import os
import pickle

import torch
import torchaudio
import numpy as np
from typing import Optional, Any
import cluster


class SpeechEncoderManager:
    def __init__(self):
        self.device = "cpu"
        self.speech_encoder_modes = [
            "vec768l12", "vec256l9", "vec256l9-onnx", "vec256l12-onnx",
            "vec768l9-onnx", "vec768l12-onnx", "hubertsoft-onnx", "hubertsoft",
            "whisper-ppg", "cnhubertlarge", "dphubert", "whisper-ppg-large",
            "wavlmbase+"
        ]
        self.speech_encoder = None
        self.speech_encoder_object = None
        self.cluster_infer_ratio = 0
        self.feature_retrieval = False
        self.spk2id = {}
        self.cluster_model = None
        self.big_npy = None
        self.now_spk_id = -1
        self.unit_interpolate_mode = 'left'

    def initialize(self, speech_encoder: str, cluster_infer_ratio: float = 0,
                   feature_retrieval: bool = False, spk2id: dict = {},
                   cluster_model_path: str = None, target_sample: int = 16000,
                   unit_interpolate_mode: str = 'left', device="cpu", **kwargs):
        """
        Initialize the speech encoder and set various parameters.

        Args:
            speech_encoder (str): Type of speech encoder.
            cluster_infer_ratio (float): Ratio for cluster inference.
            feature_retrieval (bool): Whether to use feature retrieval.
            spk2id (dict): Mapping of speaker names to IDs.
            cluster_model_path (str): Cluster model for feature retrieval.
            target_sample (int): Target sample rate.
            unit_interpolate_mode (str): Interpolation mode for units.
            **kwargs: Additional arguments for the speech encoder.
        """

        # If speech encoder and object is None, load the speech encoder
        # if not, load the speech encoder only if the speech encoder is different

        if self.speech_encoder is None or self.speech_encoder != speech_encoder:
            self.speech_encoder = speech_encoder
            self.speech_encoder_object = self.get_speech_encoder(self.speech_encoder, device, **kwargs)

            print(f"Initialized speech encoder: {speech_encoder} on {device}.")

        self.device = device
        self.cluster_infer_ratio = cluster_infer_ratio
        self.feature_retrieval = feature_retrieval
        self.spk2id = spk2id
        if cluster_model_path is not None and os.path.exists(cluster_model_path):
            if self.feature_retrieval:
                with open(cluster_model_path, "rb") as f:
                    self.cluster_model = pickle.load(f)
                self.big_npy = None
                self.now_spk_id = -1
            else:
                self.cluster_model = cluster.get_cluster_model(cluster_model_path)
        else:
            self.cluster_model = None
        self.target_sample = target_sample
        self.unit_interpolate_mode = unit_interpolate_mode
        self.audio16k_resample_transform = torchaudio.transforms.Resample(target_sample, 16000).to(self.device)

    def get_speech_encoder(self, speech_encoder, device=None, **kwargs):
        if speech_encoder == "vec768l12":
            from vencoder.ContentVec768L12 import ContentVec768L12
            speech_encoder_object = ContentVec768L12(device=device)
        elif speech_encoder == "vec256l9":
            from vencoder.ContentVec256L9 import ContentVec256L9
            speech_encoder_object = ContentVec256L9(device=device)
        elif speech_encoder == "vec256l9-onnx":
            from vencoder.ContentVec256L9_Onnx import ContentVec256L9_Onnx
            speech_encoder_object = ContentVec256L9_Onnx(device=device)
        elif speech_encoder == "vec256l12-onnx":
            from vencoder.ContentVec256L12_Onnx import ContentVec256L12_Onnx
            speech_encoder_object = ContentVec256L12_Onnx(device=device)
        elif speech_encoder == "vec768l9-onnx":
            from vencoder.ContentVec768L9_Onnx import ContentVec768L9_Onnx
            speech_encoder_object = ContentVec768L9_Onnx(device=device)
        elif speech_encoder == "vec768l12-onnx":
            from vencoder.ContentVec768L12_Onnx import ContentVec768L12_Onnx
            speech_encoder_object = ContentVec768L12_Onnx(device=device)
        elif speech_encoder == "hubertsoft-onnx":
            from vencoder.HubertSoft_Onnx import HubertSoft_Onnx
            speech_encoder_object = HubertSoft_Onnx(device=device)
        elif speech_encoder == "hubertsoft":
            from vencoder.HubertSoft import HubertSoft
            speech_encoder_object = HubertSoft(device=device)
        elif speech_encoder == "whisper-ppg":
            from vencoder.WhisperPPG import WhisperPPG
            speech_encoder_object = WhisperPPG(device=device)
        elif speech_encoder == "cnhubertlarge":
            from vencoder.CNHubertLarge import CNHubertLarge
            speech_encoder_object = CNHubertLarge(device=device)
        elif speech_encoder == "dphubert":
            from vencoder.DPHubert import DPHubert
            speech_encoder_object = DPHubert(device=device)
        elif speech_encoder == "whisper-ppg-large":
            from vencoder.WhisperPPGLarge import WhisperPPGLarge
            speech_encoder_object = WhisperPPGLarge(device=device)
        elif speech_encoder == "wavlmbase+":
            from vencoder.WavLMBasePlus import WavLMBasePlus
            speech_encoder_object = WavLMBasePlus(device=device)
        else:
            raise Exception(f"Unsupported speech encoder: {speech_encoder}, available: {self.speech_encoder_modes}")
        return speech_encoder_object

    def encode(self, wav16k: np.ndarray, speaker: Optional[str] = None) -> torch.Tensor:
        """
        Encode the input audio and apply cluster inference if enabled.

        Args:
            wav16k (np.ndarray): Input audio waveform at 16 kHz.
            speaker (Optional[str]): Speaker identifier for cluster inference.

        Returns:
            torch.Tensor: Encoded speech features.
        """
        if self.speech_encoder_object is None:
            raise Exception("Speech encoder not initialized. Call initialize() first.")

        # Encode the audio
        c = self.speech_encoder_object.encoder(wav16k)

        # Apply cluster inference if enabled
        if self.cluster_infer_ratio != 0 and self.cluster_model is not None:
            if self.feature_retrieval:
                c = self._apply_feature_retrieval(c, speaker)
            else:
                c = self._apply_cluster_inference(c, speaker)

        return c

    def _apply_feature_retrieval(self, c: torch.Tensor, speaker: str) -> torch.Tensor:
        """Apply feature retrieval for cluster inference."""
        speaker_id = self._get_speaker_id(speaker)
        feature_index = self.cluster_model[speaker_id]
        feat_np = np.ascontiguousarray(c.transpose(0, 1).cpu().numpy())

        if self.big_npy is None or self.now_spk_id != speaker_id:
            self.big_npy = feature_index.reconstruct_n(0, feature_index.ntotal)
            self.now_spk_id = speaker_id

        print("Starting feature retrieval...")
        score, ix = feature_index.search(feat_np, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        c = self.cluster_infer_ratio * npy + (1 - self.cluster_infer_ratio) * feat_np
        c = torch.FloatTensor(c).to(self.device).transpose(0, 1)
        print("End feature retrieval...")
        return c

    def _apply_cluster_inference(self, c: torch.Tensor, speaker: str) -> torch.Tensor:
        """Apply regular cluster inference."""
        cluster_c = cluster.get_cluster_center_result(self.cluster_model, c.cpu().numpy().T, speaker).T
        cluster_c = torch.FloatTensor(cluster_c).to(self.device)
        return self.cluster_infer_ratio * cluster_c + (1 - self.cluster_infer_ratio) * c

    def _get_speaker_id(self, speaker: str) -> int:
        """Get speaker ID from speaker name or raise an error."""
        speaker_id = self.spk2id.get(speaker)
        if not speaker_id and isinstance(speaker, int):
            if len(self.spk2id) >= speaker:
                speaker_id = speaker
        if speaker_id is None:
            raise RuntimeError("The name you entered is not in the speaker list!")
        return speaker_id