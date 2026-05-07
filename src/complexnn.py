from typing import List, Optional, Union

from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import torch.distributed as dist


torch.manual_seed(1)


class InstanceNorm(nn.Module):
    """Normalization along the last two dimensions, and the output shape is equal to that of the input.
    Input: B,C,T,F

    Args:
        feats: CxF with input B,C,T,F
    """

    def __init__(self, feats=1):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.gamma = nn.Parameter(torch.ones(feats), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(feats), requires_grad=True)

    def forward(self, inputs: Tensor):
        """
        inputs shape is (B, C, T, F)
        """
        nB, nC, nT, nF = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3).flatten(-2)  # B, T, CxF

        mean = torch.mean(inputs, dim=-1, keepdim=True)
        var = torch.mean(torch.square(inputs - mean), dim=-1, keepdim=True)

        std = torch.sqrt(var + self.eps)

        outputs = (inputs - mean) / std
        outputs = outputs * self.gamma + self.beta

        outputs = outputs.reshape(nB, nT, nC, nF)
        outputs = outputs.permute(0, 2, 1, 3)  # B,C,T,F

        return outputs


def complex_apply_mask(inputs, mask, method="C"):
    """
    inputs: B, C(r, i), T, F
    mask: B, C(r, i), T, F

    Return: feat_r, feat_i with shape B,C,T,F
    """
    mask_r, mask_i = torch.chunk(mask, 2, dim=1)
    feat_r, feat_i = torch.chunk(inputs, 2, dim=1)
    if method == "E":
        mask_mags = (mask_r**2 + mask_i**2) ** 0.5
        real_phase = mask_r / (mask_mags + 1e-8)
        imag_phase = mask_i / (mask_mags + 1e-8)
        mask_phase = torch.atan2(imag_phase, real_phase)

        feat_mag = (feat_r**2 + feat_i**2 + 1e-8) ** 0.5
        feat_phs = torch.atan2(feat_i, feat_r)
        # mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * feat_mag
        est_phase = feat_phs + mask_phase
        feat_r_ = est_mags * torch.cos(est_phase)
        feat_i_ = est_mags * torch.sin(est_phase)
    elif method == "C":
        feat_r_ = feat_r * mask_r - feat_i * mask_i
        feat_i_ = feat_r * mask_i + feat_i * mask_r
    else:
        raise RuntimeError("method not supported.")

    return feat_r_, feat_i_


def complex_cat(inputs, dim: int):
    """
    inputs: a list [inp1, inp2, ...]
    dim: the axis for complex features where real part first, imag part followed
    """
    real, imag = [], []

    for data in inputs:
        r, i = torch.chunk(data, 2, dim)
        real.append(r)
        imag.append(i)

    real = torch.cat(real, dim)
    imag = torch.cat(imag, dim)

    output = torch.cat([real, imag], dim)
    return output


class Guider(nn.Module):
    """
    input b,c,t,f
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.atten_r = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel // 2,
                out_channels=out_channel // 2,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            Rearrange("b c t f -> b t f c"),
            nn.LayerNorm(out_channel // 2),
            Rearrange("b t f c -> b c t f"),
            nn.Sigmoid(),
        )
        self.atten_i = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel // 2,
                out_channels=out_channel // 2,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            Rearrange("b c t f -> b t f c"),
            nn.LayerNorm(out_channel // 2),
            Rearrange("b t f c -> b c t f"),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """
        input dim should be [B, C, T, F]
        """
        real, imag = torch.chunk(input, 2, dim=1)

        real = self.atten_r(real)
        imag = self.atten_i(imag)

        return torch.cat([real, imag], dim=1)  # B,C(ri),T,F


class ComplexPReLU(nn.Module):
    def __init__(self, num_parameters, complex_dim: int = 1):
        super().__init__()
        self.prelu_r = nn.PReLU(num_parameters=num_parameters // 2)
        self.prelu_i = nn.PReLU(num_parameters=num_parameters // 2)
        self.complex_dim = complex_dim

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_dim)
        real = self.prelu_r(real)
        imag = self.prelu_i(imag)

        out = torch.cat([real, imag], dim=self.complex_dim)
        return out


class ComplexConv1d(nn.Module):
    """
    Args:
        in_channels: contains real and image, which use different conv2d to process repectively
        causal: only padding the time dimension, left side, if causal=True, otherwise padding both
        padding: [0] for Frequency dimension,
                 [1] for Time dimension as input shape is [B, 2, F, T],
                 padding Time for the requirement that need to do convolution along the frame dimension.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        causal: bool = True,
        complex_dim=1,
    ):
        super().__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.causal = causal
        self.padding = padding
        self.complex_dim = complex_dim

        self.conv_r = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
        )

        self.conv_i = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, inputs):
        """
        inputs shape should be B, F, T
        output [B, F, T]
        """

        # * padding zeros at Time dimension as Convolution kernel may be (x, 2) where 2 for Time dimension.
        if self.causal and self.padding:
            inputs = F.pad(inputs, (self.padding, 0))
        else:
            inputs = F.pad(inputs, (self.padding, self.padding))

        # NOTE need to split the input manually, because the channel dimension of input data is combined with Frequence dimension.
        # complex_dim == 0 means the input shape is (B, T, 2xF) or (T, 2xF) where 2xF indicate (Fr,..,Fr,Fi,...Fi)
        if self.complex_dim == 0:
            real = self.conv_r(inputs)
            imag = self.conv_i(inputs)
            rr, ir = torch.chunk(real, 2, self.complex_dim)
            ri, ii = torch.chunk(imag, 2, self.complex_dim)

        else:
            # * split inputs to 2 groups along complex_dim axis
            real, imag = torch.chunk(inputs, 2, self.complex_dim)  # B, C, F, T
            rr = self.conv_r(real)
            ii = self.conv_i(imag)
            ri = self.conv_i(real)
            ir = self.conv_r(imag)

        real = rr - ii
        imag = ri + ir
        out = torch.cat([real, imag], self.complex_dim)

        return out


class ComplexConv2d(nn.Module):
    """
    Input:B,C,T,F

    Args:
        in_channels: contains real and image;
        causal: only padding the time dimension, left side, if causal=True, otherwise padding both;
        padding: (time, fbin)

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal: bool = True,
        complex_dim=1,
    ):
        super().__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.causal = causal
        self.padding = padding
        self.complex_dim = complex_dim

        self.conv_r = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            padding=(0, padding[1]),
            dilation=dilation,
            groups=groups,
        )

        self.conv_i = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            padding=(0, padding[1]),
            dilation=dilation,
            groups=groups,
        )

        nn.init.normal_(self.conv_r.weight.data, mean=0.0, std=0.05)
        nn.init.constant_(self.conv_r.bias.data, 0.0)
        nn.init.normal_(self.conv_i.weight.data, mean=0.0, std=0.05)
        nn.init.constant_(self.conv_i.bias.data, 0.0)

    def forward(self, inputs):
        """
        inputs shape should be [B, C(r,i), T, F], or [C(r,i), T, F];
        output [B, Cout, T', F']
        """

        # * padding zeros at Time dimension as Convolution kernel may be (x, 2) where 2 for Time dimension.
        if self.causal and self.padding[0] != 0:
            inputs = F.pad(
                inputs,
                [0, 0, self.padding[0], 0],
            )
        elif self.padding[0] != 0:
            inputs = F.pad(inputs, [0, 0, self.padding[0], self.padding[0]])
        else:
            inputs = inputs

        # complex_dim == 0 means the input shape is (C(r,i), ...) or (C(r,i), T)
        if self.complex_dim == 0:
            real = self.conv_r(inputs)
            imag = self.conv_i(inputs)
            rr, ir = torch.chunk(real, 2, self.complex_dim)
            ri, ii = torch.chunk(imag, 2, self.complex_dim)

        else:
            # * split inputs to 2 groups along complex_dim axis
            real, imag = torch.chunk(inputs, 2, self.complex_dim)
            rr = self.conv_r(real)
            ii = self.conv_i(imag)
            ri = self.conv_i(real)
            ir = self.conv_r(imag)

        real = rr - ii
        imag = ri + ir
        out = torch.cat([real, imag], self.complex_dim)

        return out


class ComplexGateConvTranspose2d(nn.Module):
    """Gated Transposed Conv2d
    Input: B,C,T,F
    Output: B,C',T,F'

    Args:
        in_channels: real + imag
        out_channels: real + imag
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        groups=1,
        causal=False,
        complex_dim=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.complex_dim = complex_dim
        self.causal = causal

        self.tc1 = nn.Sequential(
            ComplexConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                complex_dim=complex_dim,
                groups=groups,
                causal=causal,
            ),
            nn.Sigmoid(),
        )

        self.tc2 = ComplexConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            complex_dim=complex_dim,
            groups=groups,
            causal=causal,
        )

    def forward(self, inputs):
        mask = self.tc1(inputs)
        out = self.tc2(inputs)
        out = mask * out
        return out


class ComplexConvTranspose2d(nn.Module):
    """
    Input: B,C,T,F
    Input: B,Cout,T,Fout

    Args:
        in_channels: real + imag
        out_channels: real + imag
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        groups=1,
        causal=False,
        complex_dim=1,
    ):
        super().__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.complex_dim = complex_dim
        self.causal = causal

        self.convT_r = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )
        self.convT_i = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if self.complex_dim == 0:
            real = self.convT_r(inputs)
            imag = self.convT_i(inputs)

            rr, ir = torch.chunk(real, 2, self.complex_dim)
            ri, ii = torch.chunk(imag, 2, self.complex_dim)
        else:
            real, imag = torch.chunk(inputs, 2, self.complex_dim)
            rr, ir = self.convT_r(real), self.convT_r(imag)
            ri, ii = self.convT_i(real), self.convT_i(imag)

        real = rr - ii
        imag = ir + ri

        out = torch.cat([real, imag], self.complex_dim)

        return out


class NavieComplexLSTM(nn.Module):
    """Complex LSTM
    Args:
        input_size: F of input B,C,T,F;

    Input: B,T,F
    Output: B,T,hidden_size or projection_dim
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: Optional[bool] = False,
        projection_dim: Optional[int] = None,
        bidirectional: Optional[bool] = False,
    ):
        super().__init__()

        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2
        self.batch_first = batch_first

        self.lstm_r = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.rnn_units,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        self.lstm_i = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.rnn_units,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

        if bidirectional:
            bidirectional: int = 2
        else:
            bidirectional: int = 1

        if projection_dim is not None:
            self.projection_dim = projection_dim // 2
            self.trans_r = nn.Linear(self.rnn_units * bidirectional, self.projection_dim)
            self.trans_i = nn.Linear(self.rnn_units * bidirectional, self.projection_dim)
        else:
            self.projection_dim = None

    def flatten_parameters(self):
        self.lstm_r.flatten_parameters()
        self.lstm_i.flatten_parameters()

    def forward(self, inputs: Union[list, Tensor]) -> Tensor:
        """
        inputs: (real_l, imag_l) where each element shape is [T, B, -1] or [B, T, -1]
        output: a list (real, imag)
        """
        if isinstance(inputs, list):
            real, imag = inputs
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, -1)

        rr, (h_rr, c_rr) = self.lstm_r(real)  # B,T,H
        ir, (h_ir, c_ir) = self.lstm_r(imag)
        ri, (h_ri, c_ri) = self.lstm_i(real)
        ii, (h_ii, c_ii) = self.lstm_i(imag)

        real, imag = rr - ii, ri + ir

        if self.projection_dim is not None:
            real = self.trans_r(real)  # B,T,Proj
            imag = self.trans_i(imag)

        return torch.cat([real, imag], dim=-1)  # B,T,2xO


if __name__ == "__main__":
    # inpr = torch.randn(5, 2, 3)
    # inpi = torch.randn(5, 2, 3)
    # model = NavieComplexLSTM(6, 20)
    #
    inp = torch.ones(2, 6, 2, 3)
    model = ComplexConv2d(6, 5)
    # out = model([inpr, inpi])  # ([5,2,10], [5,2,10])
    out = model(inp)
    print(out[0].shape)
