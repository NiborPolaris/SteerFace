from typing import Any
import math
import numpy as np
import torch

import torch
from typing import Optional, Iterable


class DenoisingDiffusionProbabilisticModel(torch.nn.Module):
    def __init__(self,
                 eps_model: torch.nn.Module,
                 T: int = 1000,
                 criterion: torch.nn.Module = torch.nn.MSELoss(),
                 schedule_type: str = 'linear',
                 schedule_k: float = 1.0,
                 schedule_beta_min: float = 0.0,
                 schedule_beta_max: float = 0.1) -> None:

        super(DenoisingDiffusionProbabilisticModel, self).__init__()
        self.eps_model = eps_model
        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        betas = compute_beta_schedule(T, schedule_type,
                                      k=schedule_k, beta_min=schedule_beta_min, beta_max=schedule_beta_max)

        for k, v in precompute_schedule_constants(betas).items():
            self.register_buffer(k, v)

        # for ddim sample
        self.alphas_prev = torch.tensor([1.0]).float()
        self.alphas_prev = torch.cat((self.alphas_prev, self.alpha_bars[:-1]), 0)
        # for ddim reverse sample
        self.alphas_next = torch.tensor([0.0]).float()
        # self.alphas_next = torch.cat((self.alphas_next, self.alpha_bars[1:]), 0)  # old
        self.alphas_next = torch.cat((self.alpha_bars[1:], self.alphas_next), 0)  # 好像这个才是对称的？
        # print(self.alpha_bars)  # 从1到0

        self.T = T
        self.criterion = criterion
        self.schedule_type = schedule_type
        self.schedule_k = schedule_k

    def forward(self, x0: torch.Tensor, context=None, dropout_mask=None) -> torch.Tensor:
        # t ~ U(0, T)
        t = torch.randint(0, self.T, (x0.shape[0],)).to(x0.device)
        # eps ~ N(0, 1)
        eps = torch.randn_like(x0)

        # get mean and standard deviation of p(x_t|x_0)
        mean = self.sqrt_alpha_bars[t, None, None, None] * x0
        sd = self.sqrt_one_minus_alpha_bars[t, None, None, None]

        # sample from p(x_t|x_0)
        x_t = mean + sd * eps

        return self.criterion(eps, self.eps_model(x_t, t, context, dropout_mask))

    def sample(self, n_samples, size, x_T: torch.Tensor = None, context=None, dropout_mask=None) -> torch.Tensor:
        # if initial noise is not provided then sample it
        # style test
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()
        # this samples accordingly to Algorithm 2
        self.eval()
        # torch.manual_seed(41)
        with torch.no_grad():
            # pbar = tqdm(reversed(range(0, self.T)), total=self.T)
            # pbar.set_description("DDPM Sampling")
            # for i in pbar:
            for i in reversed(range(0, self.T)):
                z = torch.randn(n_samples, *size).cuda() if i > 1 else 0
                if i == 999:
                    print(z.mean())
                t = torch.tensor(i).repeat(n_samples).cuda()
                eps = self.eps_model(x_t, t, context, dropout_mask)
                x_t = self.sqrt_alphas_inv[i] * (x_t - eps * self.one_minus_alphas_over_sqrt_one_minus_alpha_bars[i]) + \
                      self.sigmas[i] * z
        self.train()
        return x_t

    def clip(self, tensor, min_value=None, max_value=None):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            return np.clip(tensor, min_value, max_value)
        elif tensor_format == "pt":
            return torch.clamp(tensor, min_value, max_value)

    def update_sample_mask(self, mask_rate, x_t):
        emb_rate, buffer_rate = mask_rate

        bs, ch = x_t.shape[0], 256
        emb_mask = torch.bernoulli((1 - emb_rate) * torch.ones(bs, ch))
        emb_mask = emb_mask.expand(bs, ch).to(x_t.device)

        buffer_mask = torch.bernoulli((1 - buffer_rate) * torch.ones(bs, ch))
        buffer_mask = buffer_mask.expand(bs, ch).to(x_t.device)

        sample_mask = [emb_mask, buffer_mask]
        return sample_mask

    def sample_ddim(self, n_samples, size, x_T: torch.Tensor = None, context=None,
                    dropout_mask=None, eta=0, ddim_step=50) -> torch.Tensor:
        # if initial noise is not provided then sample it
        # style test
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        x_T_array = [x_t.cpu().detach().numpy()]
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding
        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"
        # TODO: 这是采样时使用的DDIM

        skip = self.T // ddim_step
        print("skip: %d" % skip)
        self.eval()

        cfg_drop_rate = None  # 采样阶段用的手动dropout，默认应为None

        with torch.no_grad():
            for i in reversed(range(0, self.T, skip)):

                t = torch.tensor(i).repeat(n_samples).cuda()

                # TODO: 不全程插入条件上下文
                # TODO: dropout_mask仅在训练时用到

                # assert len(context) == 2
                # feat_emb, physical_cond = context
                # cfg_w = 0.5  # 5
                # cfg_w = [1 - 0.04 * i for i in range(25)] + [0] + [0.04 * i for i in range(25)]

                # # TODO: TC
                # t_switch = 500  # 切换的step，step前CFG buffer，step后CFG ID
                # max_cfg_phy, max_cfg_feat = 0.5, 0.5  # CFG的基准权重
                # # max_drop_phy, max_drop_feat = 0.5, 0.5
                #
                # # step-wise权重，线性递增递减
                # if 1000 > i >= t_switch:
                #     cfg_w = max_cfg_phy * ((i - t_switch) / (1000 - t_switch))
                #     # drop_phy = 0.
                #     # drop_feat = max_drop_feat * ((i - t_switch) / (1000 - t_switch))
                #
                # elif 0 <= i < t_switch:
                #     cfg_w = max_cfg_feat * ((t_switch - i) / t_switch)
                #     # drop_phy = max_drop_phy * ((t_switch - i) / t_switch)
                #     # drop_feat = 0.
                #
                # elif i >= 1000:
                #     cfg_w = max_cfg_phy
                #     # drop_phy, drop_feat = 0., max_drop_feat
                # else:
                #     cfg_w = max_cfg_feat
                #     # drop_phy, drop_feat = max_drop_phy, 0.
                #
                # # cfg_drop_rate = [drop_feat, drop_phy]
                # # print(cfg_w, cfg_drop_rate)
                #
                # if i >= t_switch:
                #     eps_unconditional = self.eps_model(x_t, t, [context[0], None], dropout_mask, cfg_drop_rate)
                #     eps_conditional = self.eps_model(x_t, t, context, dropout_mask, cfg_drop_rate)
                #     model_output = (1 + cfg_w) * eps_conditional - cfg_w * eps_unconditional
                # else:
                #     eps_unconditional = self.eps_model(x_t, t, [None, context[1]], dropout_mask, cfg_drop_rate)
                #     eps_conditional = self.eps_model(x_t, t, context, dropout_mask, cfg_drop_rate)
                #     model_output = (1 + cfg_w) * eps_conditional - cfg_w * eps_unconditional

                # TODO: CFG buffer
                # eps_unconditional = self.eps_model(x_t, t, [context[0], None], dropout_mask)
                # eps_conditional = self.eps_model(x_t, t, context, dropout_mask)
                # model_output = (1 + cfg_w) * eps_conditional - cfg_w * eps_unconditional

                # TODO: CFG id
                # eps_unconditional = self.eps_model(x_t, t, [None, context[1]], dropout_mask)
                # eps_conditional = self.eps_model(x_t, t, context, dropout_mask)
                # model_output = (1 + cfg_w) * eps_conditional - cfg_w * eps_unconditional

                # t_start = 1000
                # t_end = 0
                #
                # if i >= t_start:
                #     scale_phy, scale_emb = 1, 0
                # elif i <= t_end:
                #     scale_phy, scale_emb = 0, 1
                # else:
                #     alpha = (i - t_end) / (t_start - t_end)
                #     scale_phy = alpha  # scale_phy 从 1 逐渐减小到 0
                #     scale_emb = 1 - alpha  # scale_emb 从 0 逐渐增大到 1
                #
                # # if i >= 500:
                # #     scale_phy, scale_emb = 1, 0
                # # else:
                # #     scale_phy, scale_emb = 0, 1
                #
                # eps_phy_cond = self.eps_model(x_t, t, [None, physical_cond], dropout_mask)
                # eps_emb_cond = self.eps_model(x_t, t, [feat_emb, None], dropout_mask)
                # eps_uncond = self.eps_model(x_t, t, [None, None], dropout_mask)
                #
                # model_output = (1 + scale_phy) * eps_phy_cond + (1 + scale_emb) * eps_emb_cond - (
                #         scale_phy + scale_emb) * eps_uncond
                # model_output = model_output / (1 + scale_phy + scale_emb)

                # model_output = self.eps_model(x_t, t, context, dropout_mask, cfg_drop_rate)
                model_output = self.eps_model(x_t, t, context)

                # 1. get previous step value (=t-1)
                prev_timestep = i - skip
                # 2. compute alphas, betas
                alpha_prod_t = self.alpha_bars[i]
                alpha_prod_t_prev = self.alphas_prev[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0).cuda()
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev
                # 3. compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                pred_original_sample = (x_t - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                # 4. Clip "predicted x_0"
                # pred_original_sample = self.clip(pred_original_sample, -1, 1)

                # 5. compute variance: "sigma_t(η)" -> see formula (16)
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
                std_dev_t = eta * variance ** (0.5)

                # the model_output is always re-derived from the clipped x_0 in Glide
                model_output = (x_t - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

                # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * model_output

                # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                x_t = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

                if eta > 0:
                    device = model_output.device if torch.is_tensor(model_output) else "cpu"
                    noise = torch.randn(n_samples, *size).cuda()
                    variance = std_dev_t * noise
                    if not torch.is_tensor(model_output):
                        variance = variance.numpy()
                    x_t = x_t + variance

                x_T_array.append(x_t.cpu().detach().numpy())

        self.train()

        return x_t, x_T_array

    def sample_ddim_reverse(self, x_start, n_samples, size=None, x_T: torch.Tensor = None, context=None,
                            dropout_mask=None, eta=0, ddim_step=50) -> torch.Tensor:

        x_t = x_T if x_T is not None else x_start.cuda()
        skip = self.T // ddim_step
        # print("skip: %d" % skip)
        self.eval()

        x_T_array = [x_start.cpu().detach().numpy()]

        # print(context.shape)

        with torch.no_grad():
            # reverse + reverse = not reverse
            for i in range(0, self.T, skip):
                t = torch.tensor(i).repeat(n_samples).cuda()
                model_output = self.eps_model(x_t, t, context, dropout_mask)
                # 1. get previous step value (=t + 1)
                next_timestep = i + skip
                # 2. compute alphas, betas
                alpha_prod_t = self.alpha_bars[i]
                alpha_prod_t_next = self.alphas_next[next_timestep] if next_timestep < self.T else torch.tensor(
                    0.0).cuda()
                beta_prod_t = 1 - alpha_prod_t
                # 3. compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                pred_original_sample = (x_t - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                # the model_output is always re-derived from the clipped x_0 in Glide
                model_output = (x_t - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

                # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                pred_sample_direction = (1 - alpha_prod_t_next) ** (0.5) * model_output

                # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                x_t = alpha_prod_t_next ** (0.5) * pred_original_sample + pred_sample_direction

                x_T_array.append(x_t.cpu().detach().numpy())

        self.train()

        return x_t, x_T_array

    def sample_classifier_free_guided(self, n_samples, size, context, guidance_scale,
                                      x_T: torch.Tensor = None) -> torch.Tensor:
        # CLASSIFIER-FREE DIFFUSION GUIDANCE: https://arxiv.org/pdf/2207.12598.pdf

        # if initial noise is not provided then sample it
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        if type(guidance_scale) is float:
            guidance_scale = [guidance_scale] * self.T

        self.eval()
        with torch.no_grad():

            # pbar = tqdm(reversed(range(0, self.T)), total=self.T)
            # pbar.set_description(f"DDPM Classifier-Free Guided Sampling")

            # for i in pbar:
            for i in reversed(range(0, self.T)):
                z = torch.randn(n_samples, *size).cuda() if i > 1 else 0
                t = torch.tensor(i).repeat(n_samples).cuda()

                eps_unconditional = self.eps_model(x_t, t, context=None, dropout_mask=None)
                eps_conditional = self.eps_model(x_t, t, context, dropout_mask=None)

                eps = (1 - guidance_scale[i]) * eps_unconditional + guidance_scale[i] * eps_conditional

                x_t = self.sqrt_alphas_inv[i] * (x_t - eps * self.one_minus_alphas_over_sqrt_one_minus_alpha_bars[i]) + \
                      self.sigmas[i] * z

        self.train()
        return x_t

    def sample_and_get_step_results(self, n_samples, size, x_T: torch.Tensor = None, context=None,
                                    dropout_mask=None, result_steps=None) -> torch.Tensor:
        # if initial noise is not provided then sample it
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        result_xs = {0: x_t.clone()}

        # this samples accordingly to Algorithm 2
        self.eval()
        with torch.no_grad():

            # pbar = tqdm(reversed(range(0, self.T)), total=self.T)
            # pbar.set_description("DDPM Sampling")

            # for i in pbar:
            for i in reversed(range(0, self.T)):
                z = torch.randn(n_samples, *size).cuda() if i > 1 else 0
                t = torch.tensor(i).repeat(n_samples).cuda()

                eps = self.eps_model(x_t, t, context, dropout_mask)
                x_t = self.sqrt_alphas_inv[i] * (x_t - eps * self.one_minus_alphas_over_sqrt_one_minus_alpha_bars[i]) + \
                      self.sigmas[i] * z

                if result_steps is not None and i + 1 in result_steps:
                    result_xs[i] = x_t.clone()

        result_xs[self.T] = x_t

        self.train()
        return result_xs

    @staticmethod
    def sample_prior(n_samples, size):
        # torch.manual_seed(41)
        # print('fixed seed: 41')
        return torch.randn(n_samples, *size)


def compute_beta_schedule(
        T: int, schedule_type: str = 'linear', k: float = 1.0,
        beta_min: float = None, beta_max: float = None) -> torch.Tensor:
    if schedule_type.lower() == 'linear':
        scale = 1000 / T
        beta_1 = scale * 0.0001
        beta_T = scale * 0.02
        return torch.linspace(beta_1, beta_T, T, dtype=torch.float32)

    elif schedule_type.lower() in ['cosine', 'cosine_warped']:
        # custom modification to cosine schedule -> warped cosine schedule
        # (this is equivalent to original cosine schedule if k=1 and beta_min=0.0)

        s = 0.008
        beta_min = 0.0 if schedule_type.lower() == 'cosine' else beta_min
        k = 1 if schedule_type.lower() == 'cosine' else k

        return betas_for_alpha_bar(
            T, lambda t: math.cos(math.pi / 2 * (t + s) / (1 + s) ** k) ** 2,
            beta_min=beta_min, beta_max=beta_max
        )

    raise NotImplementedError


def betas_for_alpha_bar(T, alpha_bar, beta_min=0.0, beta_max=1.0):
    betas = []
    for i in range(T):
        t1 = i / T
        t2 = (i + 1) / T
        betas.append(min(max(1 - alpha_bar(t2) / alpha_bar(t1), beta_min), beta_max))
    return torch.tensor(betas).float()


def precompute_schedule_constants(betas: torch.Tensor):
    alphas = 1 - betas
    sqrt_alphas_inv = 1 / alphas.sqrt()
    sigmas = betas.sqrt()
    alpha_bars = torch.cumsum(torch.log(alphas), dim=0).exp()
    sqrt_alpha_bars = alpha_bars.sqrt()
    sqrt_one_minus_alpha_bars = (1 - alpha_bars).sqrt()
    one_minus_alphas_over_sqrt_one_minus_alpha_bars = (1 - alphas) / sqrt_one_minus_alpha_bars
    return {
        "betas": betas,
        "alphas": alphas,
        "sigmas": sigmas,
        "sqrt_alphas_inv": sqrt_alphas_inv,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "one_minus_alphas_over_sqrt_one_minus_alpha_bars": one_minus_alphas_over_sqrt_one_minus_alpha_bars,
    }


if __name__ == '__main__':
    # from unet_3dg import PhysicalConditionalUNet
    #
    # eps_model = PhysicalConditionalUNet(is_context_conditional=True, context_dropout_probability=0.25)
    eps_model = None
    ddpm = DenoisingDiffusionProbabilisticModel(eps_model=eps_model)

    # x = torch.rand((64, 3, 32, 32))
    # physical_cond = torch.rand((64, 9, 128, 128)).float()
    # emb = torch.rand((64, 512)).float()
    #
    # loss = ddpm(x, [physical_cond, emb])
    # print(loss)
