import gc
import json
from typing import Optional, Any, Dict, Union, Tuple

import numpy as np
import torch

import utils
from utils import HParams, InferHParams
from manager import *


class InferManager:
    def __init__(self):
        self.device: str = "cpu"
        self.hps_ms: Optional[Union[HParams, InferHParams]] = None
        self.spk2id: Optional[Dict[str, int]] = None
        self.SE: Optional[str] = None

        self.target_sample: Optional[int] = None
        self.hop_size: Optional[int] = None

        self.f0: f0.f0Manager = f0.f0Manager()
        self.speech_encoder: speech_encoder.SpeechEncoderManager = (
            speech_encoder.SpeechEncoderManager()
        )
        self.post_processing: post_processing.PostProcessingManager = (
            post_processing.PostProcessingManager()
        )
        self.svc: svc.SVCManager = svc.SVCManager()

    def load_config(self, config_path: str):
        self.hps_ms = utils.get_hparams_from_file(config_path, True)
        self.spk2id = self.hps_ms.spk
        self.SE = self.hps_ms.model.speech_encoder
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length

    def config_loaded(self) -> bool:
        return self.hps_ms is not None

    def load_f0(self, f0: int, thres: float, trans: int, device: str):
        f0_model = self.f0.f0_modes[f0]
        self.f0.initialize(
            f0_model, self.hop_size, self.target_sample, device, thres, trans
        )

    def load_speech_encoder(
        self,
        speech_encoder: str,
        cluster_infer_ratio: float,
        feature_retrieval: bool,
        spk2id: Dict[str, int],
        cluter_model_path: str,
        device: str,
    ):
        speech_encoder_model = speech_encoder
        self.speech_encoder.initialize(
            speech_encoder_model,
            cluster_infer_ratio,
            feature_retrieval,
            spk2id,
            cluter_model_path,
            target_sample=self.target_sample,
            device=device,
        )

    def load_svc(
        self,
        svc: int,
        model_path: str,
        threshold: float,
        use_spk_mix: bool,
        pad_seconds: float,
        clip_seconds: float,
        lg_num: float,
        lgr_num: float,
        device: str,
    ):
        svc_model = self.svc.svc_modes[svc]
        self.svc.initialize(
            svc_model,
            model_path,
            self.hps_ms,
            threshold,
            use_spk_mix,
            pad_seconds,
            clip_seconds,
            lg_num,
            lgr_num,
            device,
        )

    def load_post_processor(
        self,
        post_processing: int,
        model_path: str,
        config_path: str,
        device: str = "cpu",
    ):
        post_processor = self.post_processing.post_processing_modes[post_processing]
        self.post_processing.initialize(post_processor, model_path, config_path, device)

    def load_model(self, common: Dict[str, Any], path: Dict[str, Any], device: str):
        f0 = common["f0"]
        cr_threshold = common["cr_threshold"]
        transposition = common["pitch_shift"]

        speech_encoder = path["speech_encoder"]
        cluster_model_path = path["cluster_model_path"]
        cluster_infer_ratio = common["cluster_infer_ratio"]

        svc = common["svc"]
        model_path = path["model_path"]
        threshold = common["silence_threshold"]
        use_spk_mix = common["use_spk_mix"]
        feature_retrieval = common["retrieval"]
        pad_seconds = common["pad_seconds"]
        clip_seconds = common["chunk_seconds"]
        lg_num = common["linear_gradient"]
        lgr_num = common["linear_gradient_retain"]

        post_processing = common["post_processing"]
        pp_model_path = path["pp_model_path"]
        pp_config_path = path["pp_config_path"]

        self.load_f0(f0, cr_threshold, transposition, device)
        self.load_speech_encoder(
            speech_encoder,
            cluster_infer_ratio,
            feature_retrieval,
            self.spk2id,
            cluster_model_path,
            device,
        )
        self.load_svc(
            svc,
            model_path,
            threshold,
            use_spk_mix,
            pad_seconds,
            clip_seconds,
            lg_num,
            lgr_num,
            device,
        )
        self.load_post_processor(post_processing, pp_model_path, pp_config_path, device)
        self.svc.load_components(self.f0, self.speech_encoder, self.post_processing)
        self.clear_vram()

    def check_loaded(self) -> bool:
        return (
            self.f0.f0_predictor_object is not None
            and self.speech_encoder.speech_encoder is not None
            and self.svc.svc is not None
        )

    def get_f0(self, raw_audio_path: str) -> Tuple[torch.Tensor, np.ndarray, int, int]:
        f0, wav, target_sr, hop_size = self.svc.get_f0(raw_audio_path)  # f0: [1, T]
        return f0, wav, target_sr, hop_size

    def f0_to_wav(self, f0: torch.Tensor, original_wav: np.ndarray) -> np.ndarray:
        wav, sr = self.svc.f0_to_wav(f0, original_wav)
        return wav  # [S]

    def infer(
        self,
        raw_audio_path: str,
        spk: str,
        slice_db: float,
        auto_predict_f0: bool,
        noise_scale: float,
        use_spk_mix: bool,
        loudness_envelope_adjustment: float,
        enhancer_adaptive_key: int,
        k_step: int,
        second_encoding: bool,
        use_volume: bool,
    ) -> np.ndarray:
        wav = self.svc.infer_slice(
            raw_audio_path,
            spk,
            slice_db,
            auto_predict_f0,
            noise_scale,
            use_spk_mix,
            loudness_envelope_adjustment,
            enhancer_adaptive_key,
            k_step,
            second_encoding,
            use_volume,
        )
        return wav

    def infer_with_f0(
        self,
        raw_audio_path: str,
        f0: np.ndarray,
        spk: str,
        slice_db: float,
        auto_predict_f0: bool,
        noise_scale: float,
        use_spk_mix: bool,
        loudness_envelope_adjustment: float,
        enhancer_adaptive_key:int,
        k_step: int,
        second_encoding: bool,
        use_volume: bool,
    ) -> np.ndarray:
        wav = self.svc.infer_slice_with_f0(
            raw_audio_path,
            f0,
            spk,
            slice_db,
            auto_predict_f0,
            noise_scale,
            use_spk_mix,
            loudness_envelope_adjustment,
            enhancer_adaptive_key,
            k_step,
            second_encoding,
            use_volume,
        )
        return wav

    def clear_vram(self):
        torch.cuda.empty_cache()
        gc.collect()
