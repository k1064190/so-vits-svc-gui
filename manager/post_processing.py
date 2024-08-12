from typing import Optional, Any

import torch

from manager.speech_encoder import SpeechEncoderManager
from modules.post_processing import NSFHifiGAN, ShallowDiffusion


class PostProcessingManager:
    def __init__(self):
        self.post_processing_modes = ["none", "NSF-HifiGAN", "shallow_diffusion"]
        self.post_processing: Optional[str] = None
        self.post_processing_object: Optional[Any] = None

        self.device: str = "cpu"
        self.model_path: Optional[str] = None
        self.config_path: Optional[str] = None

    def initialize(
        self,
        post_processing: str,
        model_path: str,
        config_path: str,
        device: str = "cpu",
    ):
        if (
            self.post_processing_object is None
            or post_processing != self.post_processing
            or model_path != self.model_path
            or config_path != self.config_path
            or device != self.device
        ):
            self.post_processing = post_processing
            self.post_processing_object = self.get_pp(
                post_processing, model_path, config_path
            )

    def get_pp(
        self,
        post_processing: str,
        model_path: str,
        config_path: str,
        device: str = "cpu",
    ):
        if post_processing == "none":
            return None
        elif post_processing == "NSF-HifiGAN":
            return NSFHifiGAN()
        elif post_processing == "shallow_diffusion":
            return ShallowDiffusion(model_path, config_path, device)
        else:
            raise Exception(
                f"Unsupported post processing: {post_processing}, available: {self.post_processing_modes}"
            )

    def process(
        self,
        audio: torch.Tensor,
        speech_encoder: SpeechEncoderManager,
        f0: torch.Tensor,
        c: torch.Tensor,
        vol: torch.Tensor,
        target_sample: int = 44100,
        **kwargs,
    ):
        enhancer_adaptive_key = kwargs.get("enhancer_adaptive_key", 0)
        if self.post_processing == 1:  # NSF-HifiGAN
            hop_length = kwargs.get("hop_length", 512)
            audio = self.post_processing_object.process(
                audio[None, :],
                target_sample,
                f0[:, :, None],
                hop_length,
                adaptive_key=enhancer_adaptive_key,
            )
        elif self.post_processing == 2:  # shallow_diffusion
            sid = kwargs.get("sid", 0)
            k_step = kwargs.get("k_step")
            second_encoding = kwargs.get("second_encoding", False)

            if second_encoding:
                c = speech_encoder.encode(
                    audio, None, c.shape[-1], disable_cluster=True
                )
            audio_mel = self.post_processing_object.get_vocoder.extract(
                audio[None, :], target_sample
            )

            f0 = f0[:, :, None]
            c = c.transpose(-1, -2)
            audio = self.post_processing_object.process(
                c, f0, vol, sid, audio_mel, k_step, second_encoding
            )
        else:  # None
            pass

        return audio
