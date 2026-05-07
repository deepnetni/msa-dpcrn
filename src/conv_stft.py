from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window


class STFT(nn.Module):
    """Compute FFT based on given nframe, nhop;
    the nFFT is an exponential multiple of 2 if not given.

    return: B,2,T,F
    """

    def __init__(
        self,
        nframe: int = 512,
        nhop: int = 128,
        nfft: Optional[int] = None,
        win: Union[str, np.ndarray] = "hann",
        center: bool = True,
        compress: float = 1.0,
    ):
        super().__init__()

        self.nframe = nframe
        self.nhop = nhop
        self.pad = nframe // 2 if center else 0
        self.compress_factor = compress

        if nfft is None:
            # * rounding up to an exponential multiple of 2
            self.nfft = int(2 ** np.ceil(np.log2(nframe)))
        else:
            self.nfft = nfft

        kernel, window = self.init_conv_stft_kernels(win, False)
        inv_kernel, _ = self.init_conv_stft_kernels(win, True)

        self.register_buffer("weight", kernel)
        self.register_buffer("inv_weight", inv_kernel)
        self.register_buffer("window", window)
        self.register_buffer("enframe", torch.eye(nframe)[:, None, :])

    def nLen(self, nlen: Union[List, int]) -> torch.Tensor:
        len_list = [nlen] if isinstance(nlen, int) else nlen

        L = [(l // self.nhop) * self.nhop for l in len_list]
        return torch.tensor(L[0]) if isinstance(nlen, int) else torch.tensor(L)

    def init_conv_stft_kernels(self, win=Union[str, np.ndarray], inverse=False):
        if isinstance(win, str):
            if win == "hann sqrt":
                window = get_window(
                    "hann", self.nframe, fftbins=True
                )  # fftbins=True, win is not symmetric

                window = np.sqrt(window)
            else:
                window = get_window(win, self.nframe, fftbins=True)
        elif isinstance(win, np.ndarray):
            window = win
        else:
            raise RuntimeError(f"{type(win)} is not supported.")

        N = self.nfft

        # * the fourier_baisis is nframe, N//2 + 1
        # [[ W_N^0x0, W_N^0x1, ..., W_N^0x(N-1) ]
        #  [ W_N^1x0, W_N^1x1, ..., W_N^1x(N-1) ]
        #  [ W_N^2x0, W_N^2x1, ..., W_N^2x(N-1) ]]
        fourier_basis = np.fft.rfft(np.eye(N))[: self.nframe]
        # print(fourier_basis.shape)  # 400, 257
        # * (nframe, nfft // 2 + 1)
        kernel_r, kernel_i = np.real(fourier_basis), np.imag(fourier_basis)

        # * reshape to (2 x (nfft // 2 + 1), nframe)
        kernel = np.concatenate([kernel_r, kernel_i], axis=1).T
        # print(kernel.shape)         # (514, 400)

        if inverse:
            # * A dot pinv(A) = I
            kernel = np.linalg.pinv(kernel).T

        kernel = kernel * window
        # * kernel is (out_channel, inp_channel, kernel_size)
        kernel = kernel[:, None, :]  # (2 x (nfft // 2 + 1), 1, nframe)

        return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(
            window[None, :, None].astype(np.float32)
        )

    def compress(self, x):
        """
        x: b,c,t,f
        return: b,2,t,f, b,1,t,f
        """
        x2 = torch.pow(x, 2).sum(1, keepdim=True) + 1e-12
        # b,1,t,f
        if self.compress_factor == 1.0:
            mag = torch.sqrt(x2)
        else:
            x = torch.pow(x2, (self.compress_factor - 1) / 2) * x
            mag = torch.pow(x2, self.compress_factor / 2)

        # features = torch.concatenate((mag, x), axis=1)  # b,3,t,f
        return x

    def uncompress(self, x):
        x2 = torch.pow(x, 2).sum(1, keepdim=True)
        if self.compress_factor == 1.0:
            mag = torch.sqrt(x2)
        else:
            scale = torch.pow(x2, (1.0 - self.compress_factor) / (2.0 * self.compress_factor))
            x = scale * x
            mag = torch.pow(x2, 1.0 / (2.0 * self.compress_factor))

        # features = torch.concatenate((mag, x), axis=1)  # b,3,t,f
        return x

    # def compress(self, x):
    #     r, i = x.chunk(2, dim=1)
    #     x2 = x.pow(2).sum(1, keepdim=True)
    #     pha = torch.atan2(i, r)
    #     mag = torch.pow(x2, self.compress_factor / 2)
    #     com = torch.cat((mag * torch.cos(pha), mag * torch.sin(pha)), dim=1)

    #     return com

    # def uncompress(self, x):
    #     r, i = x.chunk(2, dim=1)
    #     pha = torch.atan2(i, r)
    #     x2 = x.pow(2).sum(1, keepdim=True)
    #     mag = torch.pow(x2, (1.0 / (2 * self.compress_factor)))

    #     r, i = mag * torch.cos(pha), mag * torch.sin(pha)

    #     spex = torch.cat([r, i], dim=1)
    #     return spex

    def transform(self, x: torch.Tensor):
        """
        x shape should be: [ B, 1, T ] or [ B, T ]
        return: B,2,T,F
        """
        if x.dim() == 1:
            # T, -> B,1,T
            x = x[None, None, ...]
        elif x.dim() == 2:
            # * expand shape to (:, 1, :)
            x = torch.unsqueeze(x, dim=1)

        x = F.pad(x, (self.pad, self.pad))
        # * self.weight shape is [ 2 x (nfft//2 + 1), 1, nframe ]
        out_complex = F.conv1d(x, self.weight, stride=self.nhop)

        dim = self.nfft // 2 + 1
        real = out_complex[:, :dim, :]
        imag = out_complex[:, dim:, :]

        spec = torch.stack([real, imag], dim=1).transpose(-1, -2)

        spec = self.compress(spec)

        return spec

    def inverse(self, spec: torch.Tensor):
        """
        spec: B,2,T,F
        """

        spec = self.uncompress(spec)

        r, i = spec[:, 0, ...], spec[:, 1, ...]  # B,T,F
        inputs = torch.cat([r, i], dim=-1).transpose(-1, -2)  # B,2F,T

        outputs = F.conv_transpose1d(inputs, self.inv_weight, stride=self.nhop)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.nhop)
        outputs = outputs / (coff + 1e-8)

        outputs_ = outputs[..., self.pad : -self.pad] if self.pad != 0 else outputs
        return outputs_.squeeze(1)

    def forward(self, x):
        spec = self.transform(x)
        wav = self.inverse(spec)
        return wav


class SpecFeat(nn.Module):
    def __init__(self, stft: STFT | None = None, **kwargs) -> None:
        super().__init__()
        if stft is not None:
            self.stft = stft
        elif kwargs.get("nframe", None):
            self.stft = STFT(**kwargs)
        else:
            self.stft = None

    def to_spectrum(self, inp):
        if inp.ndim < 3:
            assert self.stft
            xk = self.stft.transform(inp)
        else:  # b,c,t,f
            xk = inp

        return xk

    @staticmethod
    def wave_frames(x, nframe, nhop, padding=True):
        """
        input: B,T
        return: b,t,d
        """
        pad = nframe // 2

        x = F.pad(x, (pad, pad)) if padding else x

        N = (x.size(-1) // nhop) * nhop

        idx = torch.arange(nframe)
        idx = torch.arange(0, N - nhop, nhop).unsqueeze(-1) + idx
        return x[:, idx]

    @staticmethod
    def logPowerSpectra(xk: torch.Tensor, norm=False):
        """Compute Log-Power Spectrogram
        xk: b,c,t,f
        """
        # Log-power spectrogram (paper formula)
        pow = xk.pow(2).sum(1, keepdim=True)
        lps = 10 * torch.log10(pow + 1e-7)  # Small epsilon for stability
        return lps

    @staticmethod
    def magPhase(xk):
        """
        xk: (b,2,t,f)
        return (b,1,t,f)
        """
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()
        xk_pha = torch.atan2(xk[:, (1,), ...], xk[:, (0,), ...])
        return xk_mag, xk_pha

    @staticmethod
    def tril(xk):
        """pick tril-angle data from xk
        output shape: (D+1)xD//2
        """
        nD = xk.size(-1)
        mask = torch.tril(torch.ones(nD, nD, dtype=torch.bool, device=xk.device))
        xk_ = xk[..., mask]
        return xk_

    @staticmethod
    def corr(xk: torch.Tensor, dim, winLen=9, step=1, sub_mean=False, tril=False):
        """Compute covariance matrix across dim
        xk: b,c,t,f
        return: b,c,t,f,w,w
        """

        assert xk.ndim == 4

        nB, nC, nT, nF = xk.shape

        if sub_mean:
            mu = xk.mean(dim, keepdim=True)
            xk = xk - mu

        r, i = xk.chunk(2, dim=1)  # b,1,t,f
        xk_comp = torch.complex(r, i)

        if dim == 2:  # T
            xk_comp_p = F.pad(xk_comp, (0, 0, winLen - 1, 0))
        elif dim == 3:  # F
            xk_comp_p = F.pad(xk_comp, (winLen - 1, 0))
        else:
            raise RuntimeError(f"dim={dim} not supported.")

        # b,1,t,f,winLen
        patches = xk_comp_p.unfold(dimension=dim, size=winLen, step=step)
        patches = patches.squeeze(1)  # btf,winLen

        xcorr = torch.einsum("btfw,btfv->btfwv", patches, patches.conj())
        xcorr = xcorr / winLen

        # btfwv 2-> b2tfwv
        xcorr = torch.view_as_real(xcorr).permute(0, 5, 1, 2, 3, 4).contiguous()

        if tril:
            mask = torch.tril(torch.ones(winLen, winLen, dtype=torch.bool, device=xk.device))
            # b,t,f,w*(w+1)//2
            xcorr = xcorr[..., mask]

        return xcorr


def verify_w_librosa(nlen):
    import librosa

    nframe = 512
    nhop = 256
    nfft = 512

    inp = torch.randn(1, nlen)
    net = STFT(nframe, nhop, nfft, "hann", center=False)
    xk = net.transform(inp)
    print("xk:", xk.shape)
    out = net.inverse(xk)
    print("xk_", out.shape, net.nLen(nlen), out[:, :10])
    # print(torch.sum((inp - out) ** 2))

    np_inputs = inp.numpy().reshape(-1)
    librosa_stft = librosa.stft(  # B,F,T
        np_inputs,
        win_length=nframe,
        n_fft=nfft,
        hop_length=nhop,
        window="hann",
        center=False,
        # center=False,
    )
    print(f"libros:{librosa_stft.shape}, {xk.shape}")

    librosa_istft = librosa.istft(
        librosa_stft,
        hop_length=nhop,
        win_length=nframe,
        n_fft=nfft,
        window="hann",
        center=False,
        # center=False,
    )
    print(f"ilibrosa:{librosa_istft.shape}", librosa_istft[:10], np_inputs[:10])

    librosa_stft = librosa_stft[None, ...]  # b,f,t
    xkk = np.stack([librosa_stft.real, librosa_stft.imag], axis=1)

    xkk = xkk.transpose(0, 1, 3, 2)  # b,2,t,f
    print("xkk", xkk.shape)
    print(xk.numpy().shape, np.sum((xk.numpy() - xkk) ** 2))


def verify_w_scipy(nlen):
    import scipy.signal as signal

    nframe = 512
    nhop = 256
    nfft = 512

    inp = torch.randn(1, nlen)
    net = STFT(nframe, nhop, nfft, "hann", center=False)
    xk = net.transform(inp)
    print("xk", xk.shape)
    out = net.inverse(xk)
    print("xk_", out.shape, net.nLen(nlen))
    # print(torch.sum((inp - out) ** 2))

    np_inputs = inp.numpy().reshape(-1)
    f, t, scipy_stft = signal.stft(  # B,F,T
        np_inputs,
        fs=16000,
        window="hann",
        nperseg=nframe,
        noverlap=nhop,
        nfft=nfft,
    )
    print(f"fshape:{f.shape}, tshape:{t.shape}")
    print(f"scipy:{scipy_stft.shape}, {xk.shape}")

    # librosa_istft = librosa.istft(
    #     librosa_stft,
    #     hop_length=nhop,
    #     win_length=nframe,
    #     n_fft=nfft,
    #     window="hann",
    #     center=False,
    #     # center=False,
    # )
    # print(f"ilibrosa:{librosa_istft.shape}")

    # librosa_stft = librosa_stft[None, ...]  # b,f,t
    # xkk = np.stack([librosa_stft.real, librosa_stft.imag], axis=1)

    # xkk = xkk.transpose(0, 1, 3, 2)  # b,2,t,f
    # print("xkk", xkk.shape)
    # print(xk.numpy().shape, np.sum((xk.numpy() - xkk) ** 2))


def verify_self():
    from matplotlib import pyplot as plt

    inp = torch.randn(1, 16000)
    # net = STFT(480, 160, 480, "hann sqrt", center=False)
    # net = STFT(480, 240, 480, "hann", center=True)
    net = STFT(510, 240, 510, "hann", center=True)
    xk = net.transform(inp)
    print(xk.shape)
    out = net.inverse(xk)
    print(out.shape)
    N = out.shape[-1]
    print(torch.sum((inp[..., :N] - out).abs()))
    # plt.subplot(211)
    plt.plot(inp[0, :N], alpha=0.3)
    # plt.subplot(212)
    plt.plot(out[0], alpha=0.3)
    # plt.show()

    # out = net(inp)
    # print(torch.sum((inp[..., :N] - out) ** 2))
    # diff = inp[0, :N] - out[0]
    # plt.plot(inp[0, :N], alpha=0.3)
    # plt.plot(out[0], alpha=0.3)
    # plt.plot(diff[0])
    # plt.show()
    # plt.savefig("a.svg")


if __name__ == "__main__":
    from utils.audiolib import audioread, audiowrite

    verify_self()

    # net = STFT(512, 256, 512, "hann", compress=0.3)

    # inp, fs = audioread("/Users/deepni/Downloads/test.wav")
    # inp_ = torch.from_numpy(inp)[None, :].float()
    # out = net(inp_).numpy()
    # out = out[0]
    # audiowrite("/Users/deepni/Downloads/test_1.wav", np.stack([inp, out], axis=-1), fs)

    # nlen = 10000
    # verify_w_librosa(nlen)
    # verify_w_scipy(nlen)
    #
    # net = STFT(512, 256, center=True)
    # out = net.nLen(nlen)
    # print(out)

    # net = STFT(256, 128, win="hann sqrt", center=False)
    # inp = torch.ones(1, 16000).float()
    # xk = net.transform(inp)
    # print(xk.shape, xk[0, 0, 0, :])

    # inp = torch.randn(1, 2, 10, 5)
    # SpecFeat.corr(inp, dim=3, winLen=4)

    # inp = torch.randn(1, 2, 4, 4)
    # print(inp)
    # out = SpecFeat.tril(inp)
    # print(out.shape, out)
