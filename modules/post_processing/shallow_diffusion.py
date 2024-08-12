import os

import torch

from diffusion.unit2mel import load_model_vocoder


class ShallowDiffusion:
    def __init__(
        self,
        diffusion_model_path: str = "pretrain/diffusion.pt",
        diffusion_config_path: str = "pretrain/config.yaml",
        device: str = "cpu",
    ):
        self.device = device
        if os.path.exists(diffusion_model_path) and os.path.exists(
            diffusion_config_path
        ):
            (
                self.diffusion_model,
                self.vocoder,
                self.diffusion_args,
            ) = load_model_vocoder(
                diffusion_model_path, device, config_path=diffusion_config_path
            )

    def get_vocoder(self):
        return self.vocoder

    def process(
        self,
        c: torch.Tensor,
        f0: torch.Tensor,
        vol: torch.Tensor,
        sid: int,
        audio_mel: torch.Tensor,
        k_step: int,
    ):
        audio_mel = self.diffusion_model(
            c,
            f0,
            vol,
            spk_id=sid,
            spk_mix_dict=None,
            gt_spec=audio_mel,
            infer=True,
            infer_speedup=self.diffusion_args.infer.speedup,
            method=self.diffusion_args.infer.method,
            k_step=k_step,
        )
        audio = self.vocoder.infer(audio_mel, f0).squeeze()
        return audio
