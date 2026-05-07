import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from src.conv_stft import STFT
from src.ft_lstm import FTLSTM_RESNET
from src.complexnn import (
    ComplexConv2d,
    ComplexGateConvTranspose2d,
    InstanceNorm,
    complex_apply_mask,
    complex_cat,
)
from src.ms_cam import AFF


class MSA_DPCRN_rev(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_mic.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(1, 5),
                        padding=(0, 2),
                        stride=(1, stride[idx]),
                    ),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                    )
                )

        self.encoder_fusion = AFF(inp_channels=self.cnn_num[-1], feature_size=nbin, r=1)

        self.rnns_r = nn.ModuleList(
            [
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            ]
        )

        self.rnns_i = nn.ModuleList(
            [
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            ]
        )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        spec_store = []
        spec = specs_mic
        x = specs_mix
        for idx, (lm, lr) in enumerate(zip(self.encoder_mic, self.encoder_rel)):
            spec = lm(spec)
            x = lr(x)  # x shape [B, C, T, F]
            spec_store.append(spec)

        x = self.encoder_fusion(x, spec)
        x_r, x_i = torch.chunk(x, 2, dim=1)

        for idx, l in enumerate(self.rnns_r):
            x_r, _ = l(x_r)

        for idx, l in enumerate(self.rnns_i):
            x_i, _ = l(x_i)

        x = torch.concatenate([x_r, x_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        x = torch.concat([feat_r, feat_i], dim=1)

        feat = x

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


if __name__ == "__main__":
    stft = STFT(128, 64, 128)

    net = MSA_DPCRN_rev(
        nframe=128,
        nhop=64,
        nfft=128,
        cnn_num=[16, 32, 64],
        stride=[2, 2, 1],
        rnn_hidden_num=64,
    )
    mic = torch.ones(1, 160000)
    ref = torch.ones(1, 160000)
    xk_mic = stft.transform(mic)
    xk_ref = stft.transform(ref)

    with torch.no_grad():
        out = net(mic, ref)

    print(out.shape)
