import configparser
import multiprocessing as mp
import os
import sys
from itertools import repeat
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

from utils.audiolib import (
    audioread,
    audiowrite,
    activitydetector,
    check_power,
    is_clipped,
    mix_w_ser,
    mix_w_snr,
)

n_cpu = mp.cpu_count()
# np.random.seed(2)
# random.seed(2)

ref_echo_index = mp.Value("i", 0)
clean_index = mp.Value("i", 0)
noise_index = mp.Value("i", 0)


def build_audio(is_clean, params, audio_sample_length=-1):
    """Construct an audio signal from source files"""
    global clean_index, noise_index

    fs_out = params["fs"]
    silence_ms = params["silence_ms"]
    silence = np.zeros(int(fs_out * silence_ms))
    if audio_sample_length == -1:
        audio_sample_length = int(params["audio_length"] * params["fs"])

    out_audio = np.zeros(0)
    remain_length = audio_sample_length

    if is_clean:
        source_files = params["clean_flist"]
        idx_counter = clean_index
    else:
        source_files = params["noise_flist"]
        idx_counter = noise_index

    while remain_length > 0:
        with idx_counter.get_lock():
            idx = idx_counter.value
            idx_counter.value += 1

        audio, fs_inp = audioread(source_files[idx % len(source_files)])
        if fs_inp != fs_out:
            continue

        if len(audio) > remain_length:
            st = np.random.randint(0, len(audio) - remain_length)
            audio = audio[st : st + remain_length]

        if is_clipped(audio) or not check_power(audio, -30):
            continue

        out_audio = np.append(out_audio, audio)
        remain_length -= len(audio)

        # add some silence if we have not reached desired audio length
        if remain_length > 0:
            silence_len = min(remain_length, len(silence))
            out_audio = np.append(out_audio, silence[:silence_len])
            remain_length -= silence_len

    return out_audio


def fetch_audio(
    is_clean: bool,
    params: Dict,
    audio_sample_length: int = -1,
):
    """Get a audio signal, and verify that it meets the activity threshold"""
    if audio_sample_length == -1:
        audio_sample_length = int(params["audio_length"] * params["fs"])

    if is_clean:
        activity_threshold = params["clean_activity_threshold"]
    else:
        activity_threshold = params["noise_activity_threshold"]

    while True:
        audio = build_audio(is_clean, params, audio_sample_length)

        if activity_threshold == 0.0:
            break

        percactive = activitydetector(audio=audio)
        if percactive > activity_threshold:
            break

    return audio


def fetch_echo(
    params: Dict,
    ref_fname: str,
    echo_fname: str,
    audio_sample_length: int = -1,
):
    """Get ref and echo signal with given filename"""
    if audio_sample_length == -1:
        audio_sample_length = int(params["audio_length"] * params["fs"])

    ref_data, ref_fs = audioread(ref_fname)
    echo_data, echo_fs = audioread(echo_fname)

    if ref_fs != echo_fs:
        raise RuntimeError("Sample rate of ref and echo are different!")

    if ref_fs != params["fs"]:
        raise RuntimeError("Sample rate of echo and target are different!")

    ref_len, echo_len = len(ref_data), len(echo_data)
    delta_len = abs(ref_len - echo_len)
    if echo_len > ref_len:
        ref_data = np.concatenate([ref_data, np.zeros(delta_len)])
    elif ref_len > echo_len:
        echo_data = np.concatenate([echo_data, np.zeros(delta_len)])
    N = max(ref_len, echo_len)

    if N < audio_sample_length:
        pad_len = audio_sample_length - N
        ref_data = np.concatenate([ref_data, np.zeros(pad_len)])
        echo_data = np.concatenate([echo_data, np.zeros(pad_len)])
    else:  # random pick
        st = np.random.randint(0, N - audio_sample_length)
        ref_data = ref_data[st : st + audio_sample_length]
        echo_data = echo_data[st : st + audio_sample_length]

    return ref_data, echo_data


def gen_echo(params: Dict, audio_sample_length: int = -1):
    global ref_echo_index

    if audio_sample_length == -1:
        audio_sample_length = int(params["audio_length"] * params["fs"])

    ref_echo_flist = params["ref_echo_flist"]
    echo_activity_threshold = params["ref_echo_activity_threshold"]

    while True:
        with ref_echo_index.get_lock():
            idx = ref_echo_index.value
            ref_echo_index.value += 1

        ref_fname, echo_fname = ref_echo_flist[idx % len(ref_echo_flist)]
        try:
            ref_data, echo_data = fetch_echo(
                params, ref_fname, echo_fname, audio_sample_length
            )
        except:
            continue

        if len(ref_data) < audio_sample_length or len(echo_data) < audio_sample_length:
            continue

        if is_clipped(ref_data) or is_clipped(echo_data):
            continue

        if not check_power(ref_data) or not check_power(echo_data):
            continue

        if echo_activity_threshold == 0.0:
            break

        percentage_ref = activitydetector(audio=ref_data)
        percentage_echo = activitydetector(audio=echo_data)

        if (
            percentage_ref > echo_activity_threshold
            and percentage_echo > echo_activity_threshold
        ):
            break

    return ref_data, echo_data, echo_fname


def main_gen_ne(params: Dict, filenum: int):
    while True:
        snr = np.round(np.random.uniform(*params["SNR"]), 2)
        is_noisy_ne = bool(
            np.random.choice(
                [0, 1], p=[1 - params["noisy_nearend"], params["noisy_nearend"]]
            )
        )

        clean = fetch_audio(True, params)
        noise = fetch_audio(False, params)
        clean_snr, noise_snr, noisy_snr, _ = mix_w_snr(
            clean,
            noise,
            snr,
            params["target_level_lower"],
            params["target_level_upper"],
        )
        if is_clipped(clean_snr) or is_clipped(noisy_snr) or is_clipped(noise_snr):
            continue

        if is_noisy_ne:
            mic_data = noisy_snr
        else:
            mic_data = clean_snr

        break

    outdir = params["out_dir"]
    outdir = os.path.join(outdir, "NE")
    outdir = os.path.join(outdir, str(filenum))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ref_p = os.path.join(outdir, "ref.wav")
    echo_p = os.path.join(outdir, "echo.wav")
    mic_p = os.path.join(outdir, "mic.wav")
    sph_p = os.path.join(outdir, "sph.wav")

    if np.random.choice([0, 1], p=[0.5, 0.5]) == 1:
        silent = np.zeros_like(clean_snr)
    else:
        silent = np.zeros_like(clean_snr) + np.finfo(float).eps

    audiowrite(ref_p, silent, params["fs"])
    audiowrite(echo_p, silent, params["fs"])
    audiowrite(mic_p, mic_data, params["fs"])
    audiowrite(sph_p, clean_snr, params["fs"])

    return 1


def main_gen_dt(params: Dict, filenum: int):
    while True:
        ser = np.round(np.random.uniform(*params["SER"]), 2)
        snr = np.round(np.random.uniform(*params["SNR"]), 2)
        is_noisy_ne = bool(
            np.random.choice(
                [0, 1], p=[1 - params["noisy_nearend"], params["noisy_nearend"]]
            )
        )

        clean = fetch_audio(True, params)
        noise = fetch_audio(False, params)
        clean_snr, noise_snr, noisy_snr, _ = mix_w_snr(
            clean,
            noise,
            snr,
            params["target_level_lower"],
            params["target_level_upper"],
        )
        if is_clipped(clean_snr) or is_clipped(noisy_snr) or is_clipped(noise_snr):
            continue

        ref_data, echo_data, echo_path = gen_echo(params)

        if is_noisy_ne:
            ne_data, echo_data, mic_data, gain = mix_w_ser(noisy_snr, echo_data, ser)
            ne_data = clean_snr * gain
        else:
            ne_data, echo_data, mic_data, _ = mix_w_ser(clean_snr, echo_data, ser)

        if is_clipped(ne_data) or is_clipped(echo_data) or is_clipped(mic_data):
            continue

        break

    outdir = params["out_dir"]
    outdir = os.path.join(outdir, "DT")
    outdir = os.path.join(outdir, str(filenum))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ref_p = os.path.join(outdir, "ref.wav")
    echo_p = os.path.join(outdir, "echo.wav")
    mic_p = os.path.join(outdir, "mic.wav")
    sph_p = os.path.join(outdir, "sph.wav")

    audiowrite(ref_p, ref_data, params["fs"])
    audiowrite(echo_p, echo_data, params["fs"])
    audiowrite(mic_p, mic_data, params["fs"])
    audiowrite(sph_p, ne_data, params["fs"])

    _, echo_fname = os.path.split(echo_path)
    echo_fname, _ = os.path.splitext(echo_fname)
    echo_p = os.path.join(outdir, echo_fname + ".txt")
    f = open(echo_p, "w")
    f.close()

    return 1


def main_aug(clean_list, noise_list, ref_echo_list, cfg_p="./template/augment.cfg"):
    if not os.path.exists(cfg_p):
        raise RuntimeError(f"{cfg_p}")
    params = read_config(cfg_p)

    params["clean_flist"] = clean_list
    params["noise_flist"] = noise_list
    params["ref_echo_flist"] = ref_echo_list

    mp.freeze_support()
    p = mp.Pool(processes=30)
    dt_files_num = params["dt_total_files"]
    ne_files_num = params["ne_total_files"]

    out = {}
    out["dt"] = list(
        p.starmap(
            main_gen_dt,
            tqdm(
                zip(repeat(params), range(dt_files_num)),
                ncols=80,
                total=dt_files_num,
            ),
        )
    )

    out["ne"] = list(
        p.starmap(
            main_gen_ne,
            tqdm(
                zip(repeat(params), range(ne_files_num)),
                ncols=80,
                total=ne_files_num,
            ),
        )
    )

    p.close()
    p.join()

    for k, v in out.items():
        v = sum(v)
        print(f"augment {k} scenario {v * params['audio_length']/3600:.2f} hours.")
    # print(len(out))


def check_params(params: Dict):
    return False


def read_config(cfg_path: str):
    params = {}
    cfg = configparser.ConfigParser()
    cfg._interpolation = configparser.ExtendedInterpolation()
    cfg.read(cfg_path, encoding="utf-8")

    # for x in cfg.items("augment"):
    #     print(x)
    cfg = cfg["augment"]
    params["fs"] = int(cfg["fs"])
    params["audio_length"] = float(cfg["audio_length"])
    params["silence_ms"] = float(cfg["silence_ms"])
    params["dt_total_hours"] = float(cfg["dt_total_hours"])
    params["ne_total_hours"] = float(cfg["ne_total_hours"])
    params["clean_activity_threshold"] = float(cfg["clean_activity_threshold"])
    params["noise_activity_threshold"] = float(cfg["noise_activity_threshold"])
    params["ref_echo_activity_threshold"] = float(cfg["ref_echo_activity_threshold"])
    params["target_level_lower"] = int(cfg["target_level_lower"])
    params["target_level_upper"] = int(cfg["target_level_upper"])
    params["noisy_nearend"] = float(cfg["noisy_nearend"])
    params["SER"] = list(map(int, cfg["SER"].split(",")))
    params["SNR"] = list(map(int, cfg["SNR"].split(",")))
    params["out_dir"] = str(cfg["out_dir"])
    params["aec_dir"] = str(cfg["aec_dir"])
    params["dns_dir"] = str(cfg["dns_dir"])

    params["dt_total_files"] = int(
        params["dt_total_hours"] * 3600 / params["audio_length"]
    )
    params["ne_total_files"] = int(
        params["ne_total_hours"] * 3600 / params["audio_length"]
    )

    return params


if __name__ == "__main__":
    main_aug([], [], [])
