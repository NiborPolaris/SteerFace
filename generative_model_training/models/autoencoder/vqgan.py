from omegaconf import OmegaConf

import sys

import torch
from models.autoencoder.quantization import VectorQuantizer2 as VectorQuantizer
from models.autoencoder.modules import Encoder, Decoder


class VQEncoderInterface(torch.nn.Module):

    def __init__(self, first_stage_config_path: str, encoder_state_dict_path: str):
        super().__init__()

        embed_dim = 3

        config = OmegaConf.load(first_stage_config_path)
        dd_config = config.params.ddconfig

        self.encoder = Encoder(**dd_config)
        self.quant_conv = torch.nn.Conv2d(dd_config["z_channels"], embed_dim, 1)

        state_dict = torch.load(encoder_state_dict_path)
        self.load_state_dict(state_dict)

    def forward(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h


class VQDecoderInterface(torch.nn.Module):

    def __init__(self, first_stage_config_path: str, decoder_state_dict_path: str):
        super().__init__()

        embed_dim = 3
        n_embed = 8192

        config = OmegaConf.load(first_stage_config_path)
        dd_config = config.params.ddconfig

        self.decoder = Decoder(**dd_config)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, dd_config["z_channels"], 1)

        state_dict = torch.load(decoder_state_dict_path)
        self.load_state_dict(state_dict)

    def forward(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


if __name__ == '__main__':
    pass
    # encoder = Encoder(double_z=False, z_channels=3, resolution=256, in_channels=3, out_ch=3, ch=128, ch_mult=(1, 2, 4),
    #                   num_res_blocks=2, attn_resolutions=[], dropout=0.0)
    # decoder = Decoder(double_z=False, z_channels=3, resolution=256, in_channels=3, out_ch=3, ch=128, ch_mult=(1, 2, 4),
    #                   num_res_blocks=2, attn_resolutions=[], dropout=0.0)
    #
    # param_size = 0
    # for param in encoder.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in encoder.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    #
    # size_all_mb = (param_size + buffer_size) / 1024 ** 2
    # print('model size: {:.3f}MB'.format(size_all_mb))
    #
    # a = torch.rand(1, 3, 128, 128)
    # b = encoder(a)
    # print(b.shape)
    # print(decoder(b).shape)
