import os
import argparse
import soundfile as sf
import numpy as np
import onnxruntime

onnxruntime.set_default_logger_severity(3)
from gcc_phat import gcc_phat


def normalize(audio, target_level=-25):
    EPS = np.finfo(float).eps

    """Normalize the signal power to the target level"""
    rms = (audio**2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def audioread(path, sub_mean=False, start=0, stop=None, target_level=None):
    """Function to read audio
    Args:
        target_level: None,int, normalize the power of data to `target_level`, default None, could be -25 dB;

    Return:
        audio, fs
    """

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))

    audio, sample_rate = sf.read(path, start=start, stop=stop)

    if sub_mean:
        audio = audio - np.mean(audio, axis=0, keepdims=True)

    if len(audio.shape) == 1:  # mono
        if target_level is not None:
            audio = normalize(audio, target_level)
    else:  # multi-channel
        audio = audio.T  # TODO check T,2 -> 2,T
        audio = audio.sum(axis=0) / audio.shape[0]
        if target_level is not None:
            audio = normalize(audio, target_level)

    return audio.astype(np.float32), sample_rate


def wav_read(mic: str, ref: str):
    d_mic, fs_1 = audioread(mic, sub_mean=True)
    d_ref, fs_2 = audioread(ref, sub_mean=True)
    assert fs_1 == fs_2

    tau, _ = gcc_phat(d_mic, d_ref, fs=fs_1, interp=1)
    tau = max(0, int((tau - 0.001) * fs_1))
    d_ref = np.concatenate([np.zeros(tau), d_ref], axis=-1, dtype=np.float32)[
        : d_mic.shape[-1]
    ]

    return d_mic, d_ref


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic", help="mic wav file", type=str)
    parser.add_argument("--ref", help="ref wav file", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    dmic, dref = wav_read(args.mic, args.ref)
    dmic = dmic[None, :]
    dref = dref[None, :]
    ort = onnxruntime.InferenceSession("model.onnx")

    inps = {"mic": dmic, "ref": dref}
    enh = ort.run(None, inps)[0]
    sf.write("out.wav", enh.squeeze(), 16000)
