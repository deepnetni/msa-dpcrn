import argparse
import os
import random
import shutil
import sys
import multiprocessing as mp
from itertools import repeat
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils.audiolib import audioread, audiowrite, mix_w_ser
from utils.parallel import Parallel

speech_dir = Path("/home/deepni/datasets/LibriSpeech/test-clean")


def concat_audio(dir: Path, min_s=6, max_s=8, tgt_s=10, tgt_hour=2, sr=16000):
    min_len, max_len = int(min_s * sr), int(max_s * sr)
    tgt_len = int(tgt_s * sr)
    tgt_num = int(tgt_hour * 3600 / tgt_s)
    audio_list = list(dir.glob("**/*.flac"))
    random.shuffle(audio_list)

    audio_stack = []

    with tqdm(total=tgt_num, ncols=80) as pbar:
        for fname in audio_list:
            audio, sr = audioread(fname)
            if len(audio) < min_len:
                continue
            elif len(audio) > max_len:
                st = np.random.randint(0, len(audio) - max_len)
                audio = audio[st : st + max_len]
                audio = np.pad(audio, pad_width=(0, tgt_len - len(audio)))
            else:
                audio = np.pad(audio, pad_width=(0, tgt_len - len(audio)))

            audio_stack.append(audio)

            pbar.update(1)
            tgt_num -= 1
            if tgt_num == 0:
                break

    audio_stack = np.array(audio_stack).flatten()
    print(audio_stack.shape)
    audiowrite("out.wav", audio_stack, sr)


def split_audio(
    lpb_fname: str, echo_fname: str, out_dir: str, split_s: int = 10, tgt_sr=16000
):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    lpb, sr_ref = audioread(lpb_fname)
    echo, sr_lpb = audioread(echo_fname)

    if sr_ref != tgt_sr:
        lpb = librosa.resample(lpb, orig_sr=sr_ref, target_sr=tgt_sr)
    if sr_lpb != tgt_sr:
        echo = librosa.resample(echo, orig_sr=sr_lpb, target_sr=tgt_sr)

    N = len(lpb)
    if len(echo) > N:
        echo = echo[:N]
    else:
        echo = np.pad(echo, pad_width=(0, N - len(echo)))

    slice_len = int(split_s * tgt_sr)
    slice_num = N // slice_len

    for i in tqdm(range(slice_num), ncols=80):
        st = i * slice_len
        lpb_slice = lpb[st : st + slice_len]
        echo_slice = echo[st : st + slice_len]

        # ref_fname = os.path.join(out_dirname, "lpb.wav")
        # lpb_fname = os.path.join(out_dirname, "echo.wav")
        ref_fname = os.path.join(out_dir, f"{i}_lpb.wav")
        echo_fname = os.path.join(out_dir, f"{i}_echo.wav")

        audiowrite(ref_fname, lpb_slice, tgt_sr)
        audiowrite(echo_fname, echo_slice, tgt_sr)


def gene_test_worker(params: Dict, filenum: int, ser: int):
    """
    Argus:
        params:
            - src_flist, list, contains [ref,echo] files
            - sph_flist, list, contain near-end wav files
            - sph_min_len, int, the minimum lenght of sph
            - out_dir, str, output dir
    """
    out_dir = Path(params["out_dir"])
    sph_list = params["sph_flist"]
    src_list = params["src_flist"]
    sph_min_len = params["sph_min_len"]

    lpb_fname, echo_fname = src_list[filenum]

    lpb, sr = audioread(lpb_fname)
    echo, _ = audioread(echo_fname)
    N = min(len(lpb), len(echo))
    lpb = lpb[:N]
    echo = echo[:N]

    while True:
        sph_fname = random.choice(sph_list)
        sph, _ = audioread(sph_fname)
        if len(sph) > sph_min_len * sr:
            break

    if len(sph) > N:
        st = np.random.randint(0, len(sph) - N)
        sph = sph[st : st + N]
    else:
        sph = np.pad(sph, pad_width=(0, N - len(sph)))

    sph_new_level, echo_new_level, mic, speech_scale = mix_w_ser(sph, lpb, ser)

    lpb_out_fname = out_dir / str(ser) / f"{filenum}_lpb.wav"
    echo_out_fname = out_dir / str(ser) / f"{filenum}_echo.wav"
    sph_out_fname = out_dir / str(ser) / f"{filenum}_sph.wav"
    mic_out_fname = out_dir / str(ser) / f"{filenum}_mic.wav"

    audiowrite(lpb_out_fname, lpb, sr)
    audiowrite(echo_out_fname, echo_new_level, sr)
    audiowrite(mic_out_fname, mic, sr)
    audiowrite(sph_out_fname, sph_new_level, sr)


def gene_test(sph_dir, ref_lpb_dir, out_dir, ser_list: List = [-10, -5, 0, 5, 10]):
    sph_dir = Path(sph_dir)
    src_dir = Path(ref_lpb_dir)
    src_flist = []
    params = {}

    sph_flist = list(sph_dir.glob("**/*.flac"))

    # print(len(sph_flist), sph_dir)

    for lpb_fname in src_dir.glob("**/*lpb.wav"):
        echo_fname = str(lpb_fname).replace("lpb", "echo")
        if os.path.exists(echo_fname):
            src_flist.append([lpb_fname, echo_fname])

    params["sph_flist"] = sph_flist
    params["src_flist"] = src_flist
    params["sph_min_len"] = 6  # seconds
    params["out_dir"] = out_dir
    print("## Generate to ", out_dir)

    worker = Parallel(nprocess=mp.cpu_count())
    for ser in ser_list:
        worker.add(
            str(ser),
            gene_test_worker,
            args=list(zip(repeat(params), range(len(src_flist)), repeat(ser))),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concat", help="concat audios", action="store_true")
    parser.add_argument("--out_h", help="concat audio len in hours", type=int)
    parser.add_argument("--inp_dir", help="input dir", type=str, default=1)

    parser.add_argument("--split", help="split audios", action="store_true")
    parser.add_argument("--split_dir", help="output dir of split audios", type=str)

    parser.add_argument("--gene", help="generate test audios", action="store_true")
    parser.add_argument(
        "--sph_dir",
        help="clean speech audios dir",
        default="/home/deepni/datasets/LibriSpeech_train/dev-clean",
    )
    parser.add_argument(
        "--ref_lpb_dir",
        help="ref and lpb wav audios dir",
        default="samples",
    )
    parser.add_argument(
        "--out_dir",
        help="output dir for test dataset",
        default="/home/deepni/datasets/ttt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.concat is True:
        # create audio to record
        inp_dir = args.inp_dir if args.inp_dir is not None else speech_dir
        concat_audio(inp_dir, 6, 8, 10, args.out_h)

    if args.split is True:
        if args.split_dir is not None:
            split_dir = args.split_dir
        else:
            split_dir = "/home/deepni/datasets/test/split"

        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

        split_audio(
            "/home/deepni/datasets/test/ref_record.wav",
            "/home/deepni/datasets/test/echo_1.wav",
            split_dir,
        )

    if args.gene is True:
        # generate double talk audios
        if os.path.exists(args.out_dir):
            shutil.rmtree(args.out_dir)

        gene_test(
            args.sph_dir, args.ref_lpb_dir, args.out_dir, ser_list=[-10, -5, 0, 5, 10]
        )
