from __future__ import annotations

from logging import getLogger
from typing import Any, Literal

import numpy as np
import torch
from cm_time import timer
from numpy import dtype, float32, ndarray
from torch import FloatTensor, Tensor

LOG = getLogger(__name__)

class f0Manager:
    def __init__(self):
        self.device = "cpu"
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.f0_modes = ["crepe", "rmvpe", "fcpe"]
        self.f0_predictor = None
        self.f0_predictor_object = None

    def initialize(self, f0_predictor, hop_size=512, target_sample=44100, device="cpu", cr_threshold=0.05, trans=0.0):
        '''

        Args:
            f0_predictor: ['crepe', 'rmvpe', 'fcpe']
            hop_size: hop size for f0 predictor(default: 512)
            target_sample: target sample rate for f0 predictor
            device: device for f0 predictor
            cr_threshold: threshold for f0 predictor(default: 0.05)(higher -> more silent frames, but precise)

        Returns: f0_predictor
        '''

        # if the arguments are different, reinitialize the f0_predictor
        if self.f0_predictor != f0_predictor or \
                self.hop_size != hop_size or self.target_sample != target_sample or self.device != device or \
                self.cr_threshold != cr_threshold:
            self.f0_predictor = f0_predictor
            self.f0_predictor_object = self.get_f0_predictor(f0_predictor=f0_predictor,
                                                              hop_length=hop_size,
                                                              f0_min=self.f0_min,
                                                              f0_max=self.f0_max,
                                                              sampling_rate=target_sample,
                                                              device=device,
                                                              threshold=cr_threshold)

            print(f"Initialized f0 predictor on {device}.")

        self.device = device

        self.hop_size = hop_size
        self.target_sample = target_sample
        self.cr_threshold = cr_threshold
        self.trans = trans

        return self.f0_predictor_object

    def get_f0_predictor(self, f0_predictor, hop_length, f0_min, f0_max, sampling_rate, **kargs):
        if f0_predictor == "crepe":
            from modules.F0Predictor.CrepeF0Predictor import CrepeF0Predictor
            f0_predictor_object = CrepeF0Predictor(hop_length=hop_length, f0_min=f0_min, f0_max=f0_max, sampling_rate=sampling_rate,
                                                   device=kargs["device"], threshold=kargs["threshold"])
        elif f0_predictor == "rmvpe":
            from modules.F0Predictor.RMVPEF0Predictor import RMVPEF0Predictor
            f0_predictor_object = RMVPEF0Predictor(hop_length=hop_length, f0_min=f0_min, f0_max=f0_max, sampling_rate=sampling_rate,
                                                   dtype=torch.float32, device=kargs["device"],
                                                   threshold=kargs["threshold"])
        elif f0_predictor == "fcpe":
            from modules.F0Predictor.FCPEF0Predictor import FCPEF0Predictor
            f0_predictor_object = FCPEF0Predictor(hop_length=hop_length, f0_min=f0_min, f0_max=f0_max, sampling_rate=sampling_rate,
                                                  dtype=torch.float32, device=kargs["device"],
                                                  threshold=kargs["threshold"])
        else:
            raise Exception(f"Unsupported f0 predictor: {f0_predictor}, available: {self.f0_modes}")
        return f0_predictor_object

    def compute_f0_uv_tran(self, wav):
        f0, uv = self.f0_predictor_object.compute_f0_uv(wav)

        f0 = torch.FloatTensor(f0).to(self.device)
        uv = torch.FloatTensor(uv).to(self.device)

        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        f0 = f0 * 2 ** (self.trans / 12)

        return f0, uv

    def compute_f0_uv(self, wav):
        f0, uv = self.f0_predictor_object.compute_f0_uv(wav)
        return f0, uv


    def f0_to_coarse(self, f0: torch.Tensor | float):
        is_torch = isinstance(f0, torch.Tensor)
        f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (self.f0_bin - 2) / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def _resize_f0(
        self, x: ndarray[Any, dtype[float32]], target_len: int
    ) -> ndarray[Any, dtype[float32]]:
        source = np.array(x)
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * target_len, len(source)) / target_len,
            np.arange(0, len(source)),
            source,
        )
        res = np.nan_to_num(target)
        return res

    def normalize_f0(
        self, f0: FloatTensor, x_mask: FloatTensor, uv: FloatTensor, random_scale=True
    ) -> FloatTensor:
        # calculate means based on x_mask
        uv_sum = torch.sum(uv, dim=1, keepdim=True)
        uv_sum[uv_sum == 0] = 9999
        means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

        if random_scale:
            factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
        else:
            factor = torch.ones(f0.shape[0], 1).to(f0.device)
        # normalize f0 based on means and factor
        f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
        if torch.isnan(f0_norm).any():
            exit(0)
        return f0_norm * x_mask


    def interpolate_f0(
        f0: ndarray[Any, dtype[float32]]
    ) -> tuple[ndarray[Any, dtype[float32]], ndarray[Any, dtype[float32]]]:
        data = np.reshape(f0, (f0.size, 1))

        vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0

        ip_data = data

        frame_number = data.size
        last_value = 0.0
        for i in range(frame_number):
            if data[i] <= 0.0:
                j = i + 1
                for j in range(i + 1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number - 1:
                    if last_value > 0.0:
                        step = (data[j] - data[i - 1]) / float(j - i)
                        for k in range(i, j):
                            ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]
                last_value = data[i]

        return ip_data[:, 0], vuv_vector[:, 0]
