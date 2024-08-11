import os

from diffusion.unit2mel import load_model_vocoder


class ShallowDiffusion:
    def __init__(
        self,
        diffusion_model_path: str = "pretrain/diffusion.pt",
        diffusion_config_path: str = "pretrain/config.yaml",
        device: str = "cpu",
    ):
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

    def process(self, data):
        return data
