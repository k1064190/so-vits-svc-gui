import json

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
        self.f0.initialize(f0, self.hps_ms.hop_size, self.hps_ms.target_sample, device, thres, trans)

    def load_speech_encoder(self, speech_encoder, device):
        self.speech_encoder.initialize(speech_encoder, device)

    def load_svc(self, svc, model_path, cluster_model_path, use_spk_mix, feature_retrieval, pad_seconds, clip_seconds, lg_num, lgr_num, device):
        self.svc.initialize(svc, model_path, self.hps_ms, cluster_model_path, use_spk_mix, feature_retrieval, pad_seconds, clip_seconds, lg_num, lgr_num, device)

    def load_model(self, f0, thres, device):
        self.f0.initialize(self.hps_ms.f0_predictor, self.hps_ms.hop_size, self.hps_ms.target_sample, self.hps_ms.device, self.hps_ms.cr_threshold)

    def infer(self):
        pass