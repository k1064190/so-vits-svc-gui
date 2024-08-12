from modules.enhancer import Enhancer


class NSFHifiGAN:
    def __init__(
        self,
        enhancer_ckpt: str = "pretrain/nsf_hifigan/model",
        device: str = "cpu",
    ):
        self.device = device
        self.enhancer = Enhancer(
            "nsf-hifigan", enhancer_ckpt, device=device
        )

    def process(self, audio, f0, target_sample, hop_length, enhancer_adaptive_key):
        audio, _ = self.enhancer.enhance(
            audio,
            target_sample,
            f0,
            hop_length,
            adaptive_key=enhancer_adaptive_key,
        )
        return audio
