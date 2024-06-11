import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.augment_mp import main_aug


class AudioDset:
    def __init__(self, dirname):
        self.dir = Path(dirname)
        self.clean_d = self.dir / "clean"
        self.noise_d = self.dir / "noise"
        self._clean_flist, self._noise_flist = self._scan()
        self.n_clean = len(self._clean_flist)
        self.n_noise = len(self._noise_flist)

        print(
            f"NS dataset has {self.n_clean:d}({self.n_clean / 120:.2f} h) clean, {self.n_noise:d}({self.n_noise / 360:.2f} h) noise wavs"
        )

    def _scan(self):
        wav_clean_l = list(map(str, self.clean_d.glob("*.wav")))
        wav_noise_l = list(map(str, self.noise_d.glob("*.wav")))

        return wav_clean_l, wav_noise_l

    @property
    def clean_flist(self):
        return self._clean_flist

    @property
    def noise_flist(self):
        return self._noise_flist


class AECDset:
    def __init__(self, dirname):
        self.dir = Path(dirname)
        self.real_d = self.dir / "real"
        self.synt_d = self.dir / "synthetic"
        self._real_flist, self._synt_flist = self._scan()
        self.n_real = len(self._real_flist)
        self.n_synt = len(self._synt_flist)

        print(
            f"AEC dataset has {self.n_real:d}({self.n_real / 360:.2f} h) real, {self.n_synt:d}({self.n_synt / 360:.2f} h) synt wavs"
        )

    def _scan(self):
        real = list(self.real_d.glob("*farend_singletalk*_mic.wav"))
        synt = list(self.synt_d.glob("echo_signal/*.wav"))
        real_flist = []
        synt_flist = []

        for f in real:
            fname = f.name.replace("mic", "lpb")
            ref_p = f.parent.joinpath(fname)
            if not ref_p.exists():
                real.remove(f)
            else:
                real_flist.append((str(ref_p), str(f)))

        for f in synt:
            ref_dir = f.parent.parent.joinpath("farend_speech")
            fname = f.name.replace("echo", "farend_speech")
            ref_p = ref_dir.joinpath(fname)

            if not ref_p.exists():
                real.remove(f)
            else:
                synt_flist.append((str(ref_p), str(f)))

        return real_flist, synt_flist

    @property
    def ref_echo_flist(self):
        return self._real_flist + self._synt_flist


if __name__ == "__main__":
    dns_dset = AudioDset("/home/deepni/datasets/DNS-Challenge/datasets")
    aec_dset = AECDset("/home/deepni/datasets/AEC-Challenge/datasets")
    cfg_p = Path("aug.cfg").resolve()
    main_aug(dns_dset.clean_flist, dns_dset.noise_flist, aec_dset.ref_echo_flist, cfg_p)
