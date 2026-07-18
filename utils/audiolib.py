import glob
import sys
from typing import Optional

import os
import wave

import librosa
import numpy as np
import soundfile as sf
import scipy.signal as sps

EPS = np.finfo(float).eps
from typing import Optional, Tuple

from scipy.signal import get_window

from collections import deque

# np.random.seed(0)


def apply_rir(audio: np.ndarray, rir: np.ndarray, N=None, norm=False, align=True):
    """apply rir operation

    :param audio: input audio with format (T,) or (T, C)
    :param rir: (N,) or (N, C)
    :param N: return N if not None
    :param norm:
    :param align: remove the delay introduce by RIR if True.
    :returns: (T, C)

    """
    # T, 1
    audio = audio[:, None] if audio.ndim < 2 else audio
    rir = rir[:, None] if rir.ndim < 2 else rir
    reverb = sps.fftconvolve(audio, rir, "full", axes=0)  # N1+N2-1

    # ! synchronize reverberant with anechoic
    if align:
        lag = np.min(np.where(np.abs(rir) >= 0.5 * np.max(np.abs(rir)))[-1])
        if N is not None:
            reverb = reverb[lag : lag + N, :]
        else:
            reverb = reverb[lag:, :]

    # ! enforce enegy conservation
    if norm:
        reverb *= np.sqrt(np.mean(audio**2) / np.mean(reverb**2))

    return reverb.squeeze()  # T, C


def clipping_solver(audio, threshold=0.99):
    """
    This function helps scale down the audio to solve the clipping issue
    returnes the scale that can be used to scale clean speech
    """
    amplitude_max = np.max(np.abs(audio))
    scale = None
    if amplitude_max <= threshold:
        return audio, scale
    else:
        scale = (amplitude_max / threshold) + EPS
        scaled_audio = audio / scale
        return scaled_audio, scale


def to_frames(
    audio: np.ndarray,
    nframe: int,
    nhop: int,
    nfft: Optional[int] = None,
    win_type: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """split audio to frames, and padding at the front and tail

    Return:
        frames: (frames_idx, frames_data)
        xk: T,F(r,i) spectrum
        L: the max point of input data

    Example:
        >>> frames, xk, L = to_frames(data, nframe, nhop)
    """

    assert nhop >= nframe // 2, f"{nhop} should be larger than half of {nframes}."
    noverlap = nframe - nhop
    n_frames = len(audio) // nhop
    L = n_frames * nhop
    # NOTE here padding `nhop` at the tail to add one frame.
    audio = np.pad(audio[:L], pad_width=(noverlap, nhop))
    n_frames += 1  # +1 for the tail padding frame
    N = len(audio)

    # frame_idx, frame_data
    slice_idx = np.arange(0, N - nframe + 1, nhop).reshape(-1, 1) + np.arange(nframe)

    frames = audio[slice_idx]

    assert n_frames == frames.shape[0], "frames number error."

    ## calculate spectrum

    if nfft is None:
        nfft = int(2 ** (np.ceil(np.log2(nframe))))

    if nhop != nframe // 2:
        window = get_window(win_type, 2 * noverlap, fftbins=True)
        window = np.concatenate(
            [window[:noverlap], np.ones(nframe - 2 * noverlap), window[-noverlap:]],
        )
    else:
        window = get_window(win_type, nframe, fftbins=True)

    # T,F(r,i)
    xk = np.fft.rfft(frames * window, n=nfft, axis=-1)
    xk = np.concatenate([xk.real, xk.imag], axis=-1)

    return frames, xk, L


def ola(audio: np.ndarray, nframe, nhop, win_type: str = "hann"):
    """restore audio from specturm by OLA method.
    Note, currently only support the conditions that the `nhop` is the 1/2 of `nfrmae`.

    Args:
        audio: T,F(r,i)

    Example:
        >>> out = ola(audio) # audio with shape T,F(r,i)
    """
    assert nhop >= nframe // 2
    assert audio.ndim == 2, "Input must at least contains 2 dimensions"

    if not isinstance(audio, np.complex128):
        r, i = np.array_split(audio, 2, axis=-1)
        audio = r + 1j * i

    noverlap = nframe - nhop
    if nhop != nframe // 2:
        window = get_window(win_type, 2 * noverlap, fftbins=True)
        window = np.concatenate(
            [window[:noverlap], np.ones(nframe - 2 * noverlap), window[-noverlap:]],
        )
    else:
        window = get_window(win_type, nframe, fftbins=True)

    div = window[-noverlap:] ** 2 + window[:noverlap] ** 2
    div = np.concatenate([div, window[noverlap:-noverlap] ** 2])  # nhop

    n_frames = audio.shape[0]
    # * +nblk is the padding length in the front when doing the stft.
    n_samples = (n_frames - 1) * nhop + nframe
    x = np.zeros((n_samples,), dtype=np.float32)

    data = np.real(np.fft.irfft(audio, axis=-1))

    # re-construct signal based on OLA
    # for i in range(n_frames):
    #     pos = np.arange(nframe) + i * nhop
    #     x[pos] += data[i, :] * window

    prev = np.zeros(nframe, dtype=np.float32)
    for i in range(n_frames):
        dd = data[i, :] * window
        current = np.concatenate([prev[-noverlap:], np.zeros(nhop)]) + dd
        prev = dd
        # divide window overlap value
        x[i * nhop : (i + 1) * nhop] = current[:nhop] / div

    # * remove the padding zeros,
    # * the last nblk values are abnormal becuase of the edge effects
    return x[noverlap:-nhop]


def check_power(audio: np.ndarray, threshold=-35) -> bool:
    """check audio is silence or not
    Return:
        True, not silence;
    """
    power = 10 * np.log10((audio**2).mean() + EPS)
    return bool(power > threshold)


def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def rms(audio, db=False):
    audio = np.asarray(audio)
    rms_value = np.sqrt(np.mean(audio**2))
    if db:
        return 20 * np.log10(rms_value + np.finfo(float).eps)
    else:
        return rms_value


def normalize(audio: np.ndarray, target_level: float = -25):
    """Normalize the signal power to the target level
    Input: T,C for stereo, T, for mono
    """
    rms = (audio**2).mean(axis=0, keepdims=True) ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio, scalar


def align_ref_to_mic(d_ref: np.ndarray, d_mic: np.ndarray, fs: int = 16000) -> np.ndarray:
    """Align the reference signal to mic signal.
    Return: the aligned ref signal.
    """
    from utils.gcc_phat import gcc_phat

    tau, _ = gcc_phat(d_mic, d_ref, fs=fs, interp=1)
    tau = max(0, int((tau - 0.001) * fs))
    d_ref = np.concatenate([np.zeros(tau), d_ref], axis=-1, dtype=np.float32)[
        : d_mic.shape[-1]
    ]
    return d_ref


def load_audio(path, start, segment_length, first_channel_only=True, target_sr=16000):
    with sf.SoundFile(path, "r") as f:
        sr = f.samplerate

        # If the audio's sample rate is different from target_sr, we need to read accordingly
        if sr != target_sr:
            start = start * sr // target_sr
            segment_length = (
                segment_length * sr + target_sr - 1
            ) // target_sr  # Fast ceil division

        f.seek(start)
        data = f.read(segment_length, dtype="float32")

    if first_channel_only and data.ndim == 2:
        data = data[:, 0]

    if sr != target_sr:
        # Ensure data is of length segment_length
        data = data[:segment_length]
        # Resample
        data = soxr.resample(data, sr, target_sr)

    return data.astype(np.float32, copy=False)


def audioread(
    path,
    sub_mean=False,
    start=0,
    stop=None,
    target_level: Optional[int] = None,
    fs: int | None = None,
) -> Tuple[np.ndarray, int]:
    """Function to read audio
    Args:
        target_level: None,int, normalize the power of data to `target_level`, default None, could be -25 dB;

    Return: audio, fs
        audio with shape (T,) for mono channel, (T, C) for stereo.
    """

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))

    # * audio shape is (T,C) for stereo or (T,) for mono
    audio, sample_rate = sf.read(path, start=start, stop=stop)

    if fs and sample_rate != fs:
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=fs)
        else:  # multi-channel
            out = []
            for i in range(audio.shape[-1]):
                dc = librosa.resample(audio[:, i], orig_sr=sample_rate, target_sr=fs)
                out.append(dc)

            audio = np.stack(out, axis=-1)

    if sub_mean:
        audio = audio - np.mean(audio, axis=0, keepdims=True)

    if target_level is not None:
        audio, _ = normalize(audio, target_level)

    return audio, sample_rate


def audiowrite(
    destpath,
    audio,
    sample_rate=16000,
    norm=False,
    target_level=-25,
    clipping_threshold=0.99,
    clip_test=False,
    fs: int | None = None,
):
    """Function to write audio
    fs -> sample_rate -> file.

    args:
        sample_rate: origianl sample rate;
        fs: resampled rate.
    """

    if clip_test:
        if is_clipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError(
                "Clipping detected in audiowrite()! " + destpath + " file not written to disk."
            )

    if fs and fs != sample_rate:
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=sample_rate)
        else:  # multi-channel
            out = []
            for i in range(audio.shape[-1]):
                dc = librosa.resample(audio[:, i], orig_sr=fs, target_sr=sample_rate)
                out.append(dc)

            audio = np.stack(out, axis=-1)

    if norm:
        audio = normalize(audio, target_level)
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio / max_amp * (clipping_threshold - EPS)

    destpath = os.path.expanduser(destpath)
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)


def activitydetector(audio, fs=16000, energy_thresh=0.13, target_level=-25):
    """Return the percentage of the time the audio signal is above an energy threshold"""

    audio = normalize(audio, target_level)
    window_size = 50  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win**2) + EPS)
        frame_energy_prob = 1.0 / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (
                1 - alpha_att
            )
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (
                1 - alpha_rel
            )

        if smoothed_energy_prob > energy_thresh:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def mix_w_snr(
    clean,
    noise,
    snr,
    target_level_lower=-35,
    target_level_upper=-15,
    target_level=-25,
    clipping_threshold=0.99,
):
    """Function to mix clean speech and noise at various SNR levels"""
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))

    # Normalizing to -25 dB FS
    clean = clean / (max(abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = (clean**2).mean() ** 0.5

    noise = noise / (max(abs(noise)) + EPS)
    noise = normalize(noise, target_level)
    rmsnoise = (noise**2).mean() ** 0.5

    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(target_level_lower, target_level_upper)
    rmsnoisy = (noisyspeech**2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        # noisy_rms_level is dBFS
        noisy_rms_level = int(
            20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS))
        )

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def mix_w_ser(speech, echo, ser, clipping_threshold=0.99):
    """Function to mix speech and echo at various SER levels
    Returns:
        speech, echo, mic, speech_scale
    """
    if len(speech) != len(echo):
        raise RuntimeError("length of speech and echo are not equal!")

    # speech = speech / (max(abs(speech)) + EPS)
    rms_speech = (speech**2).mean() ** 0.5

    # echo = echo / (max(abs(echo)) + EPS)
    rms_echo = (echo**2).mean() ** 0.5

    # Set the noise level for a given SER
    speech_scalar = rms_echo * (10 ** (ser / 20)) / (rms_speech + EPS)
    speech_new_level = speech * speech_scalar

    # Mix echo and speech
    mic_data = speech_new_level + echo

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    # TODO check if suitable
    if is_clipped(mic_data):
        mic_maxamplevel = max(abs(mic_data)) / (clipping_threshold - EPS)
        mic_data = mic_data / mic_maxamplevel
        speech_new_level = speech_new_level / mic_maxamplevel
        echo = echo / mic_maxamplevel
        speech_scalar *= 1 / mic_maxamplevel

    return speech_new_level, echo, mic_data, speech_scalar


def resampler(input_dir, target_sr=16000, ext="*.wav"):
    """Resamples the audio files in input_dir to target_sr and save"""
    files = glob.glob(f"{input_dir}/" + ext)
    for pathname in files:
        print("resample", pathname)
        try:
            audio, fs = audioread(pathname)
            audio_resampled = librosa.core.resample(audio, fs, target_sr)
            audiowrite(pathname, audio_resampled, target_sr)
        except:
            continue


def audio_segmenter(input_dir, dest_dir, segment_len=10, ext="*.wav"):
    """Segments the audio clips in dir to segment_len in secs"""
    files = glob.glob(f"{input_dir}/" + ext)
    for i in range(len(files)):
        audio, fs = audioread(files[i])

        if len(audio) > (segment_len * fs) and len(audio) % (segment_len * fs) != 0:
            audio = np.append(
                audio, audio[0 : segment_len * fs - (len(audio) % (segment_len * fs))]
            )
        if len(audio) < (segment_len * fs):
            while len(audio) < (segment_len * fs):
                audio = np.append(audio, audio)
            audio = audio[: segment_len * fs]

        num_segments = int(len(audio) / (segment_len * fs))
        audio_segments = np.split(audio, num_segments)

        basefilename = os.path.basename(files[i])
        basename, ext = os.path.splitext(basefilename)

        for j in range(len(audio_segments)):
            newname = basename + "_" + str(j) + ext
            destpath = os.path.join(dest_dir, newname)
            audiowrite(destpath, audio_segments[j], fs)


def wav2pcm(src_path, dst_path):
    """
    Examples:
        >>> wav2pcm("xx.wav", "yy.pcm")
    """
    with open(src_path, "rb") as fp:
        fp.seek(0)
        fp.read(44)  # skip the first 44 bytes wav info
        audio = np.fromfile(fp, dtype=np.int16)  # pcm is int16 type
        audio.tofile(dst_path)


def pcm2wav(src_path: str, dst_path: str, channels=1, bits=16, fs=16000):
    assert bits % 8 == 0, f"bits % 8 must == 0, while get {bits}"
    assert src_path.split(".")[-1] == "pcm" and dst_path.split(".")[-1] == "wav"

    with open(src_path, "rb") as fp:
        pcm = fp.read()

    with wave.open(dst_path, "wb") as fp:
        fp.setnchannels(channels)
        fp.setsampwidth(bits // 8)
        fp.setframerate(fs)
        fp.writeframes(pcm)
