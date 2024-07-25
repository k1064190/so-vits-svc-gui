import gc
import hashlib
import io
import json
import logging
import os
import pickle
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
import torchaudio

import cluster
import utils
from diffusion.unit2mel import load_model_vocoder
from inference import slicer
from manager.f0 import f0Manager
from manager.post_processing import PostProcessingManager
from manager.speech_encoder import SpeechEncoderManager
from models import SynthesizerTrn

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.replace("\\", "/").split("/")[-1]
                print(f"clean {f_name}")
                for wav_hash in list(data_dict.keys()):
                    if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(arr, (pad_left, pad_right), 'constant', constant_values=(0, 0))
        return padded_arr


def split_list_by_n(list_collection, n, pre=0):
    for i in range(0, len(list_collection), n):
        yield list_collection[i - pre if i - pre >= 0 else i: i + n]


class F0FilterException(Exception):
    pass


class SVCManager():
    def __init__(self):
        self.device = "cpu"
        self.svc_modes = ["so-vits-4.0"]
        self.svc = None
        self.svc_object = None
        self.hps_ms = None

    def initialize(self, svc, model_path, config, cluster_model_path, use_spk_mix, feature_retrieval,
                   pad_seconds=0.5, clip_seconds=0, lg_num=0, lgr_num=0.75, device="cpu"):
        self.device = device

        self.pad_seconds = pad_seconds
        self.clip_seconds = clip_seconds
        self.lg_num = lg_num
        self.lgr_num = lgr_num

        self.svc = svc
        self.svc_object = self.get_svc(svc, model_path, config, cluster_model_path, use_spk_mix, feature_retrieval,
                                       pad_seconds, clip_seconds, lg_num, lgr_num, device)
        self.hps_ms = self.svc_object.hps_ms
        self.spk2id = self.svc_object.spk2id
        self.speech_encoder = self.svc_object.speech_encoder


    def get_svc(self, svc, model_path, config, cluster_model_path, use_spk_mix, feature_retrieval,
                pad_seconds, clip_seconds, lg_num, lgr_num, device):
        if svc == "so-vits":
            return Svc(model_path, config, cluster_model_path, use_spk_mix, feature_retrieval,
                       pad_seconds, clip_seconds, lg_num, lgr_num, device)
        else:
            raise Exception(f"Unsupported svc: {svc}, available: {self.svc_modes}")

    def unload_model(self):
        self.svc_object.unload_model()

class Svc(object):
    def __init__(self, net_g_path,
                 config,
                 cluster_model_path=None,
                 spk_mix_enable=False,
                 feature_retrieval=False,
                pad_seconds=0.5,
                clip_seconds=0,
                lg_num=0,
                lgr_num=0.75,
                 device="cpu",
                 ):
        self.device = device

        self.pad_seconds = pad_seconds
        self.clip_seconds = clip_seconds
        self.lg_num = lg_num
        self.lgr_num = lgr_num

        self.net_g_path = net_g_path
        self.feature_retrieval = feature_retrieval
        self.net_g_ms = None
        self.hps_ms = config
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length
        self.spk2id = self.hps_ms.spk
        self.unit_interpolate_mode = self.hps_ms.data.unit_interpolate_mode if self.hps_ms.data.unit_interpolate_mode is not None else 'left'
        self.vol_embedding = self.hps_ms.model.vol_embedding if self.hps_ms.model.vol_embedding is not None else False
        self.speech_encoder = self.hps_ms.model.speech_encoder if self.hps_ms.model.speech_encoder is not None else 'vec768l12'

        # real-time
        self.last_chunk = None
        self.last_o = None
        self.chunk_len = 16000  # chunk length
        self.pre_len = 3840  # cross fade length, multiples of 640

        # if self.shallow_diffusion or self.only_diffusion:
        #     if os.path.exists(diffusion_model_path) and os.path.exists(diffusion_model_path):
        #         self.diffusion_model, self.vocoder, self.diffusion_args = load_model_vocoder(
        #             diffusion_model_path, self.device, config_path=diffusion_config_path)
        #         if self.only_diffusion:
        #             self.target_sample = self.diffusion_args.data.sampling_rate
        #             self.hop_size = self.diffusion_args.data.block_size
        #             self.spk2id = self.diffusion_args.spk
        #             self.dtype = torch.float32
        #             self.speech_encoder = self.diffusion_args.data.encoder
        #             self.unit_interpolate_mode = self.diffusion_args.data.unit_interpolate_mode if self.diffusion_args.data.unit_interpolate_mode is not None else 'left'
        #         if spk_mix_enable:
        #             self.diffusion_model.init_spkmix(len(self.spk2id))
        #     else:
        #         print("No diffusion model or config found. Shallow diffusion mode will False")
        #         self.shallow_diffusion = self.only_diffusion = False

        # load hubert and model
        self.load_model(spk_mix_enable)
        self.volume_extractor = utils.Volume_Extractor(self.hop_size)
        # self.volume_extractor = utils.Volume_Extractor(self.diffusion_args.data.block_size)

        # if self.shallow_diffusion:
        #     self.nsf_hifigan_enhance = False
        # if self.nsf_hifigan_enhance:
        #     from modules.enhancer import Enhancer
        #     self.enhancer = Enhancer('nsf-hifigan', 'pretrain/nsf_hifigan/model', device=self.device)

    def initialize(self, speech_encoder_manager, f0_manager, post_processor_manager):
        self.load_speech_encoder(speech_encoder_manager)
        self.load_f0_predictor(f0_manager)
        self.load_post_processor(post_processor_manager)

    def load_speech_encoder(self, speech_encoder_manager: SpeechEncoderManager):
        assert speech_encoder_manager.speech_encoder == self.speech_encoder
        self.speech_encoder_manager = speech_encoder_manager

    def load_f0_predictor(self, f0_predictor_manager: f0Manager):
        self.f0_predictor = f0_predictor_manager.f0_predictor
        self.f0_manager = f0_predictor_manager

    def load_post_processor(self, post_processor_manager: PostProcessingManager):
        self.post_processor = post_processor_manager
        self.post_processor_manager = post_processor_manager

    def load_model(self, spk_mix_enable=False):
        # get model configuration
        self.net_g_ms = SynthesizerTrn(
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            **self.hps_ms.model)
        _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        self.dtype = list(self.net_g_ms.parameters())[0].dtype
        if "half" in self.net_g_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.device)
        else:
            _ = self.net_g_ms.eval().to(self.device)
        if spk_mix_enable:
            self.net_g_ms.EnableCharacterMix(len(self.spk2id), self.device)

    def get_unit_f0(self, wav, speaker):
        assert self.f0_manager is not None and self.speech_encoder_manager is not None
        f0, uv = self.f0_manager.compute_f0_uv_tran(wav)

        wav = torch.from_numpy(wav).to(self.device)
        if not hasattr(self, "audio16k_resample_transform"):
            self.audio16k_resample_transform = torchaudio.transforms.Resample(self.target_sample, 16000).to(
                self.device)
        wav16k = self.audio16k_resample_transform(wav[None, :])[0]

        # speech_encoder model gets 16khz wav and returns the embedding
        # the embedding is interpolated to match the f0 length <- f0 length is important
        c = self.speech_encoder_manager.encode(wav16k, speaker)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1], self.unit_interpolate_mode)

        c = c.unsqueeze(0)
        return c, f0, uv

    def extract_f0(self, wav, sr, use_chunks=False, chunk_seconds=3.0):
        assert self.f0_predictor_object is not None
        if use_chunks:
            f0 = self.extract_f0_chunks(wav, sr, chunk_seconds)
        else:
            f0 = self.extract_f0_full(wav, sr)
        return f0

    def extract_f0_full(self, wav):
        f0, _ = self.f0_manager.compute_f0_uv(wav)
        return f0

    def extract_f0_chunks(self, wav, sr, chunk_seconds=0.5):
        # 청크 단위로 f0 추출
        chunk_size = int(chunk_seconds * sr)
        chunks = [wav[i:i + chunk_size] for i in range(0, len(wav), chunk_size)]
        f0_chunks = []
        for chunk in chunks:
            f0_chunk, _ = self.f0_manager.compute_f0_uv(chunk)
            f0_chunks.append(f0_chunk)
        return np.concatenate(f0_chunks)

    def infer(self, speaker, tran, raw_path,
              cluster_infer_ratio=0,
              auto_predict_f0=False,
              noice_scale=0.4,
              f0_filter=False,
              frame=0,
              spk_mix=False,
              loudness_envelope_adjustment=1
              ):
        torchaudio.set_audio_backend("soundfile")
        wav, sr = torchaudio.load(raw_path)
        if not hasattr(self, "audio_resample_transform") or self.audio16k_resample_transform.orig_freq != sr:
            self.audio_resample_transform = torchaudio.transforms.Resample(sr, self.target_sample)
        wav = self.audio_resample_transform(wav).numpy()[0]
        if spk_mix:
            c, f0, uv = self.get_unit_f0(wav, tran, 0, None, f0_filter)
            n_frames = f0.size(1)
            sid = speaker[:, frame:frame + n_frames].transpose(0, 1)
        else:
            speaker_id = self.spk2id.get(speaker)
            if not speaker_id and type(speaker) is int:
                if len(self.spk2id.__dict__) >= speaker:
                    speaker_id = speaker
            if speaker_id is None:
                raise RuntimeError("The name you entered is not in the speaker list!")
            sid = torch.LongTensor([int(speaker_id)]).to(self.device).unsqueeze(0)
            c, f0, uv = self.get_unit_f0(wav, tran, cluster_infer_ratio, speaker, f0_filter)
            n_frames = f0.size(1)
        c = c.to(self.dtype)
        f0 = f0.to(self.dtype)
        uv = uv.to(self.dtype)
        with torch.no_grad():
            start = time.time()
            vol = self.volume_extractor.extract(torch.FloatTensor(wav).to(self.device)[None, :])[None, :].to(
                self.device) if self.vol_embedding else None
            audio, f0 = self.net_g_ms.infer(c, f0=f0, g=sid, uv=uv, predict_f0=auto_predict_f0,
                                            noice_scale=noice_scale, vol=vol)
            audio = audio[0, 0].data.float()
            if self.dtype != torch.float32:
                c = c.to(torch.float32)
                f0 = f0.to(torch.float32)
                uv = uv.to(torch.float32)
            # audio_mel = self.vocoder.extract(audio[None, :],
            #                                  self.target_sample) if self.shallow_diffusion else None
            # if self.only_diffusion or self.shallow_diffusion:
            #     vol = self.volume_extractor.extract(audio[None, :])[None, :, None].to(
            #         self.device) if vol is None else vol[:, :, None]
            #     if self.shallow_diffusion and second_encoding:
            #         if not hasattr(self, "audio16k_resample_transform"):
            #             self.audio16k_resample_transform = torchaudio.transforms.Resample(self.target_sample,
            #                                                                               16000).to(self.device)
            #         audio16k = self.audio16k_resample_transform(audio[None, :])[0]
            #         c = self.hubert_model.encoder(audio16k)
            #         c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1], self.unit_interpolate_mode)
            #     f0 = f0[:, :, None]
            #     c = c.transpose(-1, -2)
            #     audio_mel = self.diffusion_model(
            #         c,
            #         f0,
            #         vol,
            #         spk_id=sid,
            #         spk_mix_dict=None,
            #         gt_spec=audio_mel,
            #         infer=True,
            #         infer_speedup=self.diffusion_args.infer.speedup,
            #         method=self.diffusion_args.infer.method,
            #         k_step=k_step)
            #     audio = self.vocoder.infer(audio_mel, f0).squeeze()
            # if self.nsf_hifigan_enhance:
            #     audio, _ = self.enhancer.enhance(
            #         audio[None, :],
            #         self.target_sample,
            #         f0[:, :, None],
            #         self.hps_ms.data.hop_length,
            #         adaptive_key=enhancer_adaptive_key)

            if self.post_processor is not None:
                audio = self.post_processor.process(
                    audio,
                    self.target_sample,
                    c, f0, vol,
                    self.hop_size,
                    spk_id=sid,
                )

            if loudness_envelope_adjustment != 1:
                audio = utils.change_rms(wav, self.target_sample, audio, self.target_sample,
                                         loudness_envelope_adjustment)
            use_time = time.time() - start
            print("vits use time:{}".format(use_time))
        return audio, audio.shape[-1], n_frames

    def clear_empty(self):
        # clean up vram
        torch.cuda.empty_cache()

    def unload_model(self):
        # unload model
        self.net_g_ms = self.net_g_ms.to("cpu")
        del self.net_g_ms
        gc.collect()

    def slice_inference(self,
                        raw_audio_path,
                        spk,
                        tran,
                        slice_db,
                        cluster_infer_ratio,
                        auto_predict_f0,
                        noice_scale,
                        use_spk_mix=False,
                        loudness_envelope_adjustment=1
                        ):
        if use_spk_mix:
            if len(self.spk2id) == 1:
                spk = self.spk2id.keys()[0]
                use_spk_mix = False
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        per_size = int(self.clip_seconds * audio_sr)
        lg_size = int(self.lg_num * audio_sr)
        lg_size_r = int(lg_size * self.lgr_num)
        lg_size_c_l = (lg_size - lg_size_r) // 2
        lg_size_c_r = lg_size - lg_size_r - lg_size_c_l
        lg = np.linspace(0, 1, lg_size_r) if lg_size != 0 else 0

        if use_spk_mix:
            assert len(self.spk2id) == len(spk)
            audio_length = 0
            for (slice_tag, data) in audio_data:
                aud_length = int(np.ceil(len(data) / audio_sr * self.target_sample))
                if slice_tag:
                    audio_length += aud_length // self.hop_size
                    continue
                if per_size != 0:
                    datas = split_list_by_n(data, per_size, lg_size)
                else:
                    datas = [data]
                for k, dat in enumerate(datas):
                    pad_len = int(audio_sr * self.pad_seconds)
                    per_length = int(np.ceil(len(dat) / audio_sr * self.target_sample))
                    a_length = per_length + 2 * pad_len
                    audio_length += a_length // self.hop_size
            audio_length += len(audio_data)
            spk_mix_tensor = torch.zeros(size=(len(spk), audio_length)).to(self.device)
            for i in range(len(spk)):
                last_end = None
                for mix in spk[i]:
                    if mix[3] < 0. or mix[2] < 0.:
                        raise RuntimeError("mix value must higer Than zero!")
                    begin = int(audio_length * mix[0])
                    end = int(audio_length * mix[1])
                    length = end - begin
                    if length <= 0:
                        raise RuntimeError("begin Must lower Than end!")
                    step = (mix[3] - mix[2]) / length
                    if last_end is not None:
                        if last_end != begin:
                            raise RuntimeError("[i]EndTime Must Equal [i+1]BeginTime!")
                    last_end = end
                    if step == 0.:
                        spk_mix_data = torch.zeros(length).to(self.device) + mix[2]
                    else:
                        spk_mix_data = torch.arange(mix[2], mix[3], step).to(self.device)
                    if (len(spk_mix_data) < length):
                        num_pad = length - len(spk_mix_data)
                        spk_mix_data = torch.nn.functional.pad(spk_mix_data, [0, num_pad], mode="reflect").to(
                            self.device)
                    spk_mix_tensor[i][begin:end] = spk_mix_data[:length]

            spk_mix_ten = torch.sum(spk_mix_tensor, dim=0).unsqueeze(0).to(self.device)
            # spk_mix_tensor[0][spk_mix_ten<0.001] = 1.0
            for i, x in enumerate(spk_mix_ten[0]):
                if x == 0.0:
                    spk_mix_ten[0][i] = 1.0
                    spk_mix_tensor[:, i] = 1.0 / len(spk)
            spk_mix_tensor = spk_mix_tensor / spk_mix_ten
            if not ((torch.sum(spk_mix_tensor, dim=0) - 1.) < 0.0001).all():
                raise RuntimeError("sum(spk_mix_tensor) not equal 1")
            spk = spk_mix_tensor

        global_frame = 0
        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
            # padd
            length = int(np.ceil(len(data) / audio_sr * self.target_sample))
            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(length)
                audio.extend(list(pad_array(_audio, length)))
                global_frame += length // self.hop_size
                continue
            if per_size != 0:
                datas = split_list_by_n(data, per_size, lg_size)
            else:
                datas = [data]
            for k, dat in enumerate(datas):
                per_length = int(
                    np.ceil(len(dat) / audio_sr * self.target_sample)) if self.clip_seconds != 0 else length
                if self.clip_seconds != 0:
                    print(f'###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                # padd
                pad_len = int(audio_sr * self.pad_seconds)
                dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                raw_path = io.BytesIO()
                soundfile.write(raw_path, dat, audio_sr, format="wav")
                raw_path.seek(0)
                out_audio, out_sr, out_frame = self.infer(spk, tran, raw_path,
                                                          cluster_infer_ratio=cluster_infer_ratio,
                                                          auto_predict_f0=auto_predict_f0,
                                                          noice_scale=noice_scale,
                                                          frame=global_frame,
                                                          spk_mix=use_spk_mix,
                                                          loudness_envelope_adjustment=loudness_envelope_adjustment
                                                          )
                global_frame += out_frame
                _audio = out_audio.cpu().numpy()
                pad_len = int(self.target_sample * self.pad_seconds)
                _audio = _audio[pad_len:-pad_len]
                _audio = pad_array(_audio, per_length)
                if lg_size != 0 and k != 0:
                    lg1 = audio[-(lg_size_r + lg_size_c_r):-lg_size_c_r] if self.lgr_num != 1 else audio[-lg_size:]
                    lg2 = _audio[lg_size_c_l:lg_size_c_l + lg_size_r] if self.lgr_num != 1 else _audio[0:lg_size]
                    lg_pre = lg1 * (1 - lg) + lg2 * lg
                    audio = audio[0:-(lg_size_r + lg_size_c_r)] if self.lgr_num != 1 else audio[0:-lg_size]
                    audio.extend(lg_pre)
                    _audio = _audio[lg_size_c_l + lg_size_r:] if self.lgr_num != 1 else _audio[lg_size:]
                audio.extend(list(_audio))
        return np.array(audio)

    def slice_inference_with_f0(self, raw_audio_path, f0, spk, tran, slice_db, cluster_infer_ratio,
                                auto_predict_f0, noise_scale, pad_seconds=0.5):
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

        per_size = int(self.clip_seconds * audio_sr)
        lg_size = int(self.lg_num * audio_sr)
        lg_size_r = int(lg_size * self.lgr_num)
        lg_size_c_l = (lg_size - lg_size_r) // 2
        lg_size_c_r = lg_size - lg_size_r - lg_size_c_l
        lg = np.linspace(0, 1, lg_size_r) if lg_size != 0 else 0

        audio = []
        f0_output = []
        f0_index = 0

        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(len(data))
                audio.extend(list(_audio))
                f0_output.extend([0] * (len(data) // self.hop_size))
                continue

            # 현재 청크의 f0 계산
            f0_chunk = f0[f0_index:f0_index + len(data) // self.hop_size]
            f0_index += len(data) // self.hop_size

            # clip_seconds에 따라 청크 나누기
            datas = split_list_by_n(data, per_size, lg_size)
            f0_chunks = split_list_by_n(f0_chunk, per_size // self.hop_size, lg_size // self.hop_size)

            for k, (dat, f0_dat) in enumerate(zip(datas, f0_chunks)):
                print(f'----->segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                pad_len = int(audio_sr * pad_seconds)
                dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                f0_dat = np.concatenate(
                    [np.zeros(pad_len // self.hop_size), f0_dat, np.zeros(pad_len // self.hop_size)])

                # 추론을 위한 임시 파일 생성
                raw_path = io.BytesIO()
                soundfile.write(raw_path, dat, audio_sr, format="wav")
                raw_path.seek(0)

                # 수정된 f0를 사용하여 추론
                out_audio, out_sr, out_f0 = self.infer_with_f0(
                    spk, tran, raw_path, f0_dat,
                    cluster_infer_ratio=cluster_infer_ratio,
                    auto_predict_f0=auto_predict_f0,
                    noise_scale=noise_scale
                )

                out_audio = out_audio.cpu().numpy()

                # 패딩 제거
                out_audio = out_audio[pad_len:-pad_len]
                out_f0 = out_f0[pad_len // self.hop_size: -(pad_len // self.hop_size)]

                # crossfade 적용
                if k != 0:
                    out_audio[:lg_size_r] = out_audio[:lg_size_r] * lg + audio[-lg_size_r:] * (1 - lg)
                    out_f0[:lg_size_r // self.hop_size] = out_f0[:lg_size_r // self.hop_size] * lg[
                                                                                                :lg_size_r // self.hop_size] + f0_output[
                                                                                                                               -lg_size_r // self.hop_size:] * (
                                                                      1 - lg[:lg_size_r // self.hop_size])
                    audio = audio[:-lg_size_r]
                    f0_output = f0_output[:-lg_size_r // self.hop_size]

                audio.extend(list(out_audio))
                f0_output.extend(list(out_f0))

        return np.array(audio), np.array(f0_output)

    def infer_with_f0(self, speaker, tran, raw_path, f0, cluster_infer_ratio=0, auto_predict_f0=False, noise_scale=0.4):
        # infer 메소드와 유사하지만 f0를 직접 받아 사용
        wav, sr = torchaudio.load(raw_path)
        wav = self.audio_resample_transform(wav).numpy()[0]

        c, _, uv = self.get_unit_f0(wav, tran, cluster_infer_ratio, speaker, False)

        # 제공된 f0 사용
        f0 = torch.FloatTensor(f0).unsqueeze(0).to(self.device)

        c = c.to(self.dtype)
        f0 = f0.to(self.dtype)
        uv = uv.to(self.dtype)

        with torch.no_grad():
            sid = torch.LongTensor([self.spk2id[speaker]]).to(self.device).unsqueeze(0)
            audio = self.net_g_ms.infer(c, f0=f0, g=sid, uv=uv, predict_f0=auto_predict_f0, noise_scale=noise_scale)[0][
                0].data.float()

            if self.post_processor is not None:
                vol = self.volume_extractor.extract(audio.unsqueeze(0))[0]
                audio = self.post_processor.process(
                    audio, self.target_sample, c, f0, vol,
                    self.hop_size, spk_id=sid
                )

        return audio, self.target_sample, f0.shape[1]

    def realtime_infer(self, speaker_id, f_pitch_change, input_wav_path,
                cluster_infer_ratio=0,
                auto_predict_f0=False,
                noice_scale=0.4,
                f0_filter=False):

        import maad
        audio, sr = torchaudio.load(input_wav_path)
        audio = audio.cpu().numpy()[0]
        temp_wav = io.BytesIO()
        if self.last_chunk is None:
            input_wav_path.seek(0)

            audio, sr, _ = self.infer(speaker_id, f_pitch_change, input_wav_path,
                                        cluster_infer_ratio=cluster_infer_ratio,
                                        auto_predict_f0=auto_predict_f0,
                                        noice_scale=noice_scale,
                                        f0_filter=f0_filter)

            audio = audio.cpu().numpy()
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return audio[-self.chunk_len:]
        else:
            audio = np.concatenate([self.last_chunk, audio])
            soundfile.write(temp_wav, audio, sr, format="wav")
            temp_wav.seek(0)

            audio, sr, _ = self.infer(speaker_id, f_pitch_change, temp_wav,
                                        cluster_infer_ratio=cluster_infer_ratio,
                                        auto_predict_f0=auto_predict_f0,
                                        noice_scale=noice_scale,
                                        f0_filter=f0_filter)

            audio = audio.cpu().numpy()
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return ret[self.chunk_len:2 * self.chunk_len]

