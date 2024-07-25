import gc
import json

import torch

import utils
from manager import *


class InferManager():
    def __init__(self):
        self.device = "cpu"

        self.f0 = f0.f0Manager()
        self.speech_encoder = speech_encoder.SpeechEncoderManager()
        self.post_processing = post_processing.PostProcessingManager()
        self.svc = svc.SVCManager()

    def load_config(self, config_path):
        # defer initialization until running inference.
        self.hps_ms = utils.get_hparams_from_file(config_path, True)

    def load_f0(self, f0, thres, trans, device):
        f0_model = self.f0.f0_modes[f0]
        self.f0.initialize(f0_model, self.hps_ms.hop_size, self.hps_ms.target_sample, device, thres, trans)

    def load_speech_encoder(self, speech_encoder, device):
        speech_encoder_model = self.speech_encoder.speech_encoder_modes[speech_encoder]
        self.speech_encoder.initialize(speech_encoder_model, device)

    def load_svc(self, svc, model_path, use_spk_mix, feature_retrieval, pad_seconds, clip_seconds, lg_num, lgr_num, device):
        svc_model = self.svc.svc_modes[svc]
        self.svc.initialize(svc_model, model_path, self.hps_ms, use_spk_mix, feature_retrieval, pad_seconds, clip_seconds, lg_num, lgr_num, device)

    def load_model(self, common, device):
        device = torch.device(device["device"])

        f0 = common["f0"]
        cr_threshold = common["cr_threshold"]
        transposition = common["pitch_shift"]
        self.load_f0(f0, cr_threshold, transposition, device)

        speech_encoder = common["speech_encoder"]
        self.load_speech_encoder(speech_encoder, device)

        svc = common["svc"]
        model_path = svc["model_path"]
        cluster_model_path = svc["cluster_model_path"]
        use_spk_mix = svc["use_spk_mix"]
        feature_retrieval = svc["retrieval"]
        pad_seconds = svc["pad_seconds"]
        clip_seconds = svc["clip_seconds"]
        lg_num = svc["linear_gradient"]
        lgr_num = svc["linear_gradient_retain"]
        self.load_svc(svc, model_path, cluster_model_path, use_spk_mix, feature_retrieval, pad_seconds, clip_seconds, lg_num, lgr_num, device)





    def infer(self):
        pass

    def clear_vram(self):
        torch.cuda.empty_cache()
        gc.collect()