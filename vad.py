import webrtcvad
import numpy as np
from collections import deque


class VAD:
    def __init__(self, blk_ms=30, fs=16000, level=3):
        """
        level: range from 0 to 3, while 0 is the least aggressive about filtering out non-speech
        blk_len should be 10, 20, 30ms
        """
        self.fs = fs
        self.block_len = blk_ms * fs // 1000
        self.block_bytes = int(2 * self.block_len)  # only support int16 dtype
        self.web_vad = webrtcvad.Vad()
        self.web_vad.set_mode(level)
        self.history = deque(maxlen=128)
        self.active = False
        self.data = b""

    def is_speech(self, data):
        """
        data should be (N,)
        """
        self.data += data.tobytes()

        while len(self.data) >= self.block_bytes:
            blk = self.data[: self.block_bytes]
            self.data = self.data[self.block_bytes :]

            if self.web_vad.is_speech(blk, self.fs) is True:
                self.history.append(1)
            else:
                self.history.append(0)

            active_num = 0
            for i in range(-8, 0):
                try:
                    active_num += self.history[i]
                except IndexError:
                    continue

            if not self.active:
                if active_num >= 4:
                    self.active = True
                    break
                elif (
                    len(self.history) == self.history.maxlen and sum(self.history) == 0
                ):
                    for _ in range(int(self.history.maxlen / 2)):
                        self.history.popleft()
            else:
                if active_num == 0:
                    self.active = False
                elif sum(self.history) > self.history.maxlen * 0.9:
                    for _ in range(int(self.history.maxlen / 2)):
                        self.history.popleft()

        return self.active

    def reset(self):
        self.data = b""
        self.active = False
        self.history.clear()


if __name__ == "__main__":
    import soundfile as sf

    f = "/home/ll/datasets/blind_test_set/clean/-3sybEBJmEC8P7T6LAFqoA_doubletalk_mic.wav"
    inp, fs = sf.read(f)
    inp = inp * 32768
    inp = inp.astype(np.int16)
    print(inp.dtype)
    v = VAD()

    for st in range(1, 10000, 1000):
        out = v.is_speech(inp[st : st + 1000])
        print(out)
