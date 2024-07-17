from __future__ import annotations

from logging import getLogger
from typing import Any, Literal

import numpy as np
import torch
import torchcrepe
from cm_time import timer
from numpy import dtype, float32, ndarray
from torch import FloatTensor, Tensor

LOG = getLogger(__name__)

class f0Manager:
    def __init__(self):
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.f0_modes = ["crepe", "rmvpe", "fcpe", "none"]

    def compute_f0(
            self,
            wav_numpy: ndarray[Any, dtype[float32]],
            p_len: None | int = None,
            sampling_rate: int = 44100,
            hop_length: int = 512,
            method: str = "crepe",
            f0_methods: list = [],
            **kwargs,
    ):
        with timer() as t:
            wav_numpy = wav_numpy.astype(np.float32)
            wav_numpy /= np.quantile(np.abs(wav_numpy), 0.999)
            function_name = f"compute_f0_{method}"
            if method not in f0_methods:
                raise ValueError(f"Unsupported f0 method: {method}, available: {f0_methods}")
            f0 = getattr(self, function_name)(wav_numpy, p_len, sampling_rate, hop_length, **kwargs)
        rtf = t.elapsed / (len(wav_numpy) / sampling_rate)
        LOG.info(f"F0 inference time:       {t.elapsed:.3f}s, RTF: {rtf:.3f}")
        return f0

    def compute_f0_crepe(
            self,
            wav_numpy: ndarray[Any, dtype[float32]],
            p_len: None | int = None,
            sampling_rate: int = 44100,
            hop_length: int = 512,
            device: str | torch.device = "cpu",
            model: Literal["full", "tiny"] = "full",
    ):
        audio = torch.from_numpy(wav_numpy).to(device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)

        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        # (T) -> (1, T)
        audio = audio.detach()

        pitch: Tensor = torchcrepe.predict(
            audio,
            sampling_rate,
            hop_length,
            self.f0_min,
            self.f0_max,
            model,
            batch_size=hop_length * 2,
            device=device,
            pad=True,
        )

        f0 = pitch.squeeze(0).cpu().float().numpy()
        p_len = p_len or wav_numpy.shape[0] // hop_length
        f0 = self._resize_f0(f0, p_len)
        return f0


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
