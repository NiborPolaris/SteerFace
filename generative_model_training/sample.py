import os
import sys

from typing import Any
import math
import numpy as np
import hydra
import torch
from pytorch_lightning.lite import LightningLite
import torch.nn.functional as F
import omegaconf
from omegaconf import OmegaConf, DictConfig

from hydra.utils import instantiate
from torchvision.utils import save_image
from models.autoencoder.vqgan import VQEncoderInterface, VQDecoderInterface
from models.diffusion.ddpm import DenoisingDiffusionProbabilisticModel
from utils.helpers import ensure_path_join, denormalize_to_zero_to_one
from utils.runtime_log import inject_runtime



class DiffusionSamplerLite(LightningLite):
    def run(self, cfg) -> Any:
        train_cfg_path = cfg.model_config
        train_cfg = omegaconf.OmegaConf.load(train_cfg_path)

        self.seed_everything(cfg.sampling.seed * (1 + self.global_rank))

        diffusion_model = instantiate(train_cfg)

        checkpoint_path = cfg.checkpoint.path

        weights = torch.load(checkpoint_path, map_location="cpu")
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('eps_model.', '') if 'eps_model' in k else k
            weights_dict[new_k] = v
        weights_dict_2 = {}
        for k, v in weights_dict.items():
            new_k = k.replace('module.', '') if 'module.' in k else k
            weights_dict_2[new_k] = v

        diffusion_model.load_state_dict(weights_dict_2, strict=False)  # 只加载默认参数
        diffusion_model = DenoisingDiffusionProbabilisticModel(eps_model=diffusion_model)

        diffusion_model = self.setup(diffusion_model)
        diffusion_model.eval()

        size = (3, 128, 128)
        latent_encoder = VQEncoderInterface(
            first_stage_config_path=cfg.autoencoder.config,
            encoder_state_dict_path=cfg.autoencoder.encoder
        )

        size = latent_encoder(torch.ones([1, *size])).shape[-3:]
        latent_encoder = self.setup(latent_encoder)
        latent_encoder.eval()
        latent_decoder = VQDecoderInterface(
            first_stage_config_path=cfg.autoencoder.config,
            decoder_state_dict_path=cfg.autoencoder.decoder
        )
        latent_decoder = self.setup(latent_decoder)
        latent_decoder.eval()

        assert cfg.sampling.contexts_file is not None

        contexts = torch.load(cfg.sampling.contexts_file)

        sorted_keys_values = sorted(contexts.items(), key=lambda x: int(x[0].split('/')[cfg.sampling.context_key_index]))  # 旧格式：1

        sorted_values = [value for key, value in sorted_keys_values]

        contexts = np.stack(sorted_values)

        if len(contexts) > cfg.sampling.n_contexts:  # 10000
            contexts = contexts[:cfg.sampling.n_contexts]
        contexts_norm = np.linalg.norm(contexts, axis=1)
        contexts = contexts / contexts_norm[:, np.newaxis]

        context_ids = list(i for i in range(0, contexts.shape[0]))

        samples_dir = cfg.sampling.save_dir

        def id_done(i: int) -> bool:
            p = os.path.join(samples_dir, str(i))
            return os.path.isdir(p) and len(
                [f for f in os.listdir(p) if f.endswith('.png')]) >= cfg.sampling.n_samples_per_context

        length_before = len(context_ids)
        context_ids = [i for i in context_ids if not id_done(i)]
        print(f"Skipped {length_before - len(context_ids)} context ids (already sampled).")

        context_ids = self.split_across_devices(context_ids)
        print(context_ids[:10])

        if self.global_rank == 0:
            inject_runtime(cfg)  # 获取运行时信息，加入config
            with open(ensure_path_join(f"{samples_dir}.yaml"), "w+") as f:
                OmegaConf.save(config=cfg, f=f.name)

        np.random.seed(1337)

        print(len(context_ids))

        for id_index in range(0, len(context_ids)):

            prefix = str(context_ids[id_index])
            print("sample " + prefix)
            context = torch.from_numpy(contexts[context_ids[id_index]]).float()

            context = context.repeat(cfg.sampling.batch_size, 1).cuda()

            while not isinstance(diffusion_model, DenoisingDiffusionProbabilisticModel):
                diffusion_model = diffusion_model.module

            self.perform_sampling(
                diffusion_model=diffusion_model,
                n_samples=cfg.sampling.n_samples_per_context,
                size=size,
                batch_size=cfg.sampling.batch_size,
                samples_dir=samples_dir,
                prefix=prefix,
                context=context,
                latent_encoder=latent_encoder,
                latent_decoder=latent_decoder,
                start_batch=None
            )

            sys.stdout.flush()

    @staticmethod
    def perform_sampling(
            diffusion_model, n_samples, size, batch_size, samples_dir,
            prefix: str = None, context: torch.Tensor = None, latent_encoder: torch.nn.Module = None,
            latent_decoder: torch.nn.Module = None, start_batch: torch.Tensor = None):

        n_batches = math.ceil(n_samples / batch_size)

        samples_for_grid = []

        if context is not None:
            assert prefix is not None

        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir, exist_ok=True)

        with torch.no_grad():
            for _ in range(n_batches):
                batch_samples, _ = diffusion_model.sample_ddim(batch_size, size, x_T=start_batch, context=context)
                with torch.no_grad():
                    if latent_decoder:
                        batch_samples = latent_decoder(batch_samples).cpu()

                batch_samples = denormalize_to_zero_to_one(batch_samples)
                samples_for_grid.append(batch_samples)
                samples = torch.cat(samples_for_grid, dim=0)[:n_samples]

                samples = F.interpolate(samples, size=[112, 112], mode='bilinear')
                save_path = os.path.join(samples_dir, prefix)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                for sample_index in range(len(samples)):
                    save_image(samples[sample_index], ensure_path_join(save_path, str(sample_index) + ".png"))

    def split_across_devices(self, L):
        if type(L) is int:
            L = list(range(L))

        L_per_device = [L[i::self.world_size] for i in range(self.world_size)]

        return L_per_device[self.global_rank]

@hydra.main(config_path='./configs', config_name='sample_ddim_config', version_base=None)
def sample(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = DiffusionSamplerLite(devices="auto", accelerator="auto")
    sampler.run(cfg)

if __name__ == "__main__":
    sample()
