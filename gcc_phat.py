import numpy as np
import sys
from pathlib import Path

from vad import VAD
from scipy.signal import get_window


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    """
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    """

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / (np.abs(R) + 1e-10), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau, cc


class DelayEstimator:
    def __init__(self, blk_ms=30, blk_n=4, fs=16000, interp=1):
        self.vad = VAD(blk_ms, fs)
        self.fs = fs
        self.blk_len = int(blk_ms * fs // 1000)
        self.blk_n = blk_n
        self.sblk_buf = np.zeros((blk_n, self.blk_len))
        self.rblk_buf = np.zeros((blk_n, self.blk_len))
        self.win = get_window("hann", self.blk_len * blk_n, fftbins=True)
        self.interp = interp

    def delayList(self, sig, ref):
        """
        data shape should be (N,)
        """
        N1, N2 = sig.shape[-1], ref.shape[-1]
        N = min(N1, N2)
        blk_num = N // self.blk_len
        N = self.blk_len * blk_num
        sig = sig[..., :N]
        ref = ref[..., :N]

        slices = np.arange(self.blk_len) + np.arange(0, N, self.blk_len).reshape(-1, 1)
        sblk_data = sig[slices]
        rblk_data = ref[slices]

        delay = []

        nframe = 0
        for sblk, rblk in zip(sblk_data, rblk_data):
            self.sblk_buf[0, :] = sblk
            self.rblk_buf[0, :] = rblk

            sblk_int = sblk * 32768
            sblk_int = sblk_int.astype(np.int16)

            if nframe >= self.blk_n and self.vad.is_speech(sblk_int) is True:
                sblks = self.sblk_buf.flatten()[::-1] * self.win
                rblks = self.rblk_buf.flatten()[::-1] * self.win
                tau, _ = gcc_phat(
                    sblks, rblks, self.fs, max_tau=0.1, interp=self.interp
                )
                # delay.append(int(tau * self.fs))
                delay.append(round(tau, 4))

            self.sblk_buf[1:, :] = self.sblk_buf[:-1, :]
            self.rblk_buf[1:, :] = self.rblk_buf[:-1, :]
            nframe += 1

        return np.array(delay)


if __name__ == "__main__":
    import soundfile as sf

    # refsig = np.linspace(1, 10, 10)

    # for i in range(0, 10):
    #     sig = np.concatenate((np.linspace(0, 0, i), refsig, np.linspace(0, 0, 10 - i)))
    #     offset, _ = gcc_phat(sig, refsig, fs=2)
    #     print(offset)

    # f1 = "/Users/deepni/datasets/blind_test_set/clean/-3sybEBJmEC8P7T6LAFqoA_doubletalk_mic.wav"
    # f2 = "/Users/deepni/datasets/blind_test_set/clean/-3sybEBJmEC8P7T6LAFqoA_doubletalk_lpb.wav"
    #
    f1 = "/home/deepni/datasets/howling/label/338.wav"
    f2 = "/home/deepni/datasets/howling/train/338.wav"
    mic, fs = sf.read(f1)
    ref, _ = sf.read(f2)
    obj = DelayEstimator(blk_ms=30, blk_n=50)
    out = obj.delayList(mic, ref)
    print(out.mean())
    shift = int(out.mean() * fs)

    data = np.roll(mic, -shift)
    # sf.write("/Users/deepni/datasets/out.wav", data, fs)

    tau, _ = gcc_phat(mic, ref, fs)
    data = np.roll(mic, -int(tau * fs))
    # sf.write("/Users/deepni/datasets/out1.wav", data, fs)
    print(tau)
