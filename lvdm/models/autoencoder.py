import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange

from lvdm.distributions import (
    DiagonalGaussianDistribution,
)
from lvdm.modules.networks.ae_modules import (
    Decoder,
    Encoder,
)

from utils.general_utils import instantiate_from_config


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        embed_dim,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        logdir=None,
        input_dim=4,
        test_args=None,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.test_args = test_args
        self.logdir = logdir
        if colorize_nlabels is not None:
            if not isinstance(colorize_nlabels, int):
                raise TypeError(
                    f"colorize_nlabels must be int, got {type(colorize_nlabels)}"
                )
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

    def encode(self, x, **kwargs):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, **kwargs):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if x.dim() == 5 and self.input_dim == 4:
            b, c, t, h, w = x.shape
            self.b = b
            self.t = t
            x = rearrange(x, "b c t h w -> (b t) c h w")

        return x

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = {}
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
