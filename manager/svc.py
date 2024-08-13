import gc
import hashlib
import io
import json
import logging
import os
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Union, Optional, List, Any, Tuple, Dict

import librosa
import numpy as np
import soundfile
import torch
import torchaudio

import cluster
import utils
from utils import HParams, InferHParams
from diffusion.unit2mel import load_model_vocoder
from inference import slicer
from manager.f0 import f0Manager
from manager.post_processing import PostProcessingManager
from manager.speech_encoder import SpeechEncoderManager
from models import SynthesizerTrn

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def read_temp(file_name: str) -> Dict[str, Any]:
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
                    if (
                        int(time.time()) - int(data_dict[wav_hash]["time"])
                        > 14 * 24 * 3600
                    ):
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


def write_temp(file_name: str, data: Dict[str, Any]) -> None:
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print("executing '%s' costed %.3fs" % (func.__name__, time.time() - t))
        return res

    return run


def format_wav(audio_path: Union[str, Path]) -> None:
    if Path(audio_path).suffix == ".wav":
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def get_end_file(dir_path: str, end: str) -> List[str]:
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != "."]
        dirs[:] = [d for d in dirs if d[0] != "."]
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def get_md5(content: Union[str, bytes]) -> str:
    return hashlib.new(
        "md5", content.encode() if isinstance(content, str) else content
    ).hexdigest()


def fill_a_to_b(a: List[Any], b: List[Any]) -> None:
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: List[str]) -> None:
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def pad_array(arr: np.ndarray, target_length: int) -> np.ndarray:
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(
            arr, (pad_left, pad_right), "constant", constant_values=(0, 0)
        )
        return padded_arr


def split_list_by_n(
    list_collection: List[Any], n: int, pre: int = 0
) -> List[List[Any]]:
    for i in range(0, len(list_collection), n):
        yield list_collection[i - pre if i - pre >= 0 else i : i + n]


class F0FilterException(Exception):
    pass


class Svc:
    def __init__(
        self,
        net_g_path: str,
        config: Union[HParams, InferHParams],
        threshold: float = 0.05,
        spk_mix_enable: bool = False,
        pad_seconds: float = 0.5,
        clip_seconds: float = 0,
        lg_num: float = 0,
        lgr_num: float = 0.75,
        device: str = "cpu",
    ):
        self.device = device
        self.dtype = torch.float32

        self.pad_seconds = pad_seconds
        self.clip_seconds = clip_seconds
        self.lg_num = lg_num
        self.lgr_num = lgr_num
        self.slice_db = threshold

        self.net_g_path = net_g_path
        self.net_g_ms = None
        self.hps_ms = config
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length
        self.spk2id = self.hps_ms.spk
        self.unit_interpolate_mode = (
            self.hps_ms.data.unit_interpolate_mode
            if self.hps_ms.data.unit_interpolate_mode is not None
            else "left"
        )
        self.vol_embedding = (
            self.hps_ms.model.vol_embedding
            if self.hps_ms.model.vol_embedding is not None
            else False
        )
        self.speech_encoder = (
            self.hps_ms.model.speech_encoder
            if self.hps_ms.model.speech_encoder is not None
            else "vec768l12"
        )
        self.audio_resample_transform = None
        self.audio_resample_transform_orig_freq = None

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

    def initialize(
        self,
        speech_encoder_manager: SpeechEncoderManager,
        f0_manager: f0Manager,
        post_processor_manager: PostProcessingManager,
    ) -> None:
        self.load_speech_encoder(speech_encoder_manager)
        self.load_f0_predictor(f0_manager)
        self.load_post_processor(post_processor_manager)

    def update(
        self,
        pad_seconds: float,
        clip_seconds: float,
        lg_num: float,
        lgr_num: float,
        threshold: float,
    ):
        self.pad_seconds = pad_seconds
        self.clip_seconds = clip_seconds
        self.lg_num = lg_num
        self.lgr_num = lgr_num
        self.slice_db = threshold

    def load_speech_encoder(self, speech_encoder_manager: SpeechEncoderManager) -> None:
        assert speech_encoder_manager.speech_encoder == self.speech_encoder
        self.speech_encoder_manager = speech_encoder_manager

    def load_f0_predictor(self, f0_predictor_manager: f0Manager) -> None:
        self.f0_manager = f0_predictor_manager

    def load_post_processor(
        self, post_processor_manager: PostProcessingManager
    ) -> None:
        self.post_processor_manager = post_processor_manager

    def load_model(self, spk_mix_enable: bool = False) -> None:
        # get model configuration
        self.net_g_ms = SynthesizerTrn(
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            **self.hps_ms.model,
        )
        _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        self.dtype = list(self.net_g_ms.parameters())[0].dtype
        if "half" in self.net_g_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.device)
        else:
            _ = self.net_g_ms.eval().to(self.device)
        if spk_mix_enable:
            self.net_g_ms.EnableCharacterMix(len(self.spk2id), self.device)

    def load_wav(self, raw_audio_path: str) -> Tuple[np.ndarray, int, int, int]:
        wav, sr = librosa.load(raw_audio_path, sr=None)
        if (
            not hasattr(self, "audio_resample_transform")
            or self.audio_resample_transform_orig_freq != sr
        ):
            self.audio_resample_transform = partial(
                librosa.resample, orig_sr=sr, target_sr=self.target_sample
            )
            self.audio_resample_transform_orig_freq = sr
        wav = self.audio_resample_transform(wav)
        return wav, sr, self.target_sample, self.hop_size

    def get_unit_f0(
        self, wav: np.ndarray, speaker: Optional[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.f0_manager is not None and self.speech_encoder_manager is not None
        f0, uv = self.f0_manager.compute_f0_uv_tran(wav)

        wav = torch.from_numpy(wav).to(self.device).to(self.dtype)

        # speech_encoder model gets 16khz wav and returns the embedding
        # the embedding is interpolated to match the f0 length <- f0 length is important
        c = self.speech_encoder_manager.encode(wav, speaker, f0.shape[1])

        return c, f0, uv

    def get_unit(self, wav: np.ndarray, speaker: Optional[str], f0: torch.Tensor
    ) -> torch.Tensor:
        assert self.speech_encoder_manager is not None

        wav = torch.from_numpy(wav).to(self.device).to(self.dtype)

        c = self.speech_encoder_manager.encode(wav, speaker, f0.shape[1])
        return c

    def infer(
        self,
        speaker: Union[str, torch.Tensor],
        wav: np.ndarray,
        sr: int,
        auto_predict_f0: bool = False,
        noice_scale: float = 0.4,
        frame: int = 0,
        spk_mix: bool = False,
        loudness_envelope_adjustment: float = 1,
        post_processor: PostProcessingManager = None,
        enhancer_adaptive_key: int = 0,
        k_step: int = 100,
        second_encoding: bool = False,
        use_volume: bool = True,
    ) -> Tuple[torch.Tensor, int, int]:
        if (
            not hasattr(self, "audio_resample_transform")
            or self.audio_resample_transform_orig_freq != sr
        ):
            self.audio_resample_transform = partial(
                librosa.resample, orig_sr=sr, target_sr=self.target_sample
            )
            self.audio_resample_transform_orig_freq = sr
        wav = self.audio_resample_transform(wav)
        if spk_mix:
            c, f0, uv = self.get_unit_f0(wav, None)
            n_frames = f0.size(1)
            sid = speaker[:, frame : frame + n_frames].transpose(0, 1)
        else:
            speaker_id = self.spk2id.get(speaker)
            if not speaker_id and type(speaker) is int:
                if len(self.spk2id.__dict__) >= speaker:
                    speaker_id = speaker
            if speaker_id is None:
                raise RuntimeError("The name you entered is not in the speaker list!")
            sid = torch.LongTensor([int(speaker_id)]).to(self.device).unsqueeze(0)
            c, f0, uv = self.get_unit_f0(wav, speaker)
            n_frames = f0.size(1)
        c = c.to(self.dtype)
        f0 = f0.to(self.dtype)
        uv = uv.to(self.dtype)
        with torch.no_grad():
            start = time.time()
            vol = (
                self.volume_extractor.extract(
                    torch.FloatTensor(wav).to(self.device)[None, :]
                )[None, :].to(self.device)
                if use_volume
                else None
            )
            audio, f0 = self.net_g_ms.infer(
                c,
                f0=f0,
                g=sid,
                uv=uv,
                predict_f0=auto_predict_f0,
                noice_scale=noice_scale,
                vol=vol,
            )
            audio = audio[0, 0].data.float()
            if self.dtype != torch.float32:
                c = c.to(torch.float32)
                f0 = f0.to(torch.float32)
                uv = uv.to(torch.float32)

            if not use_volume:
                vol = self.volume_extractor.extract(audio[None, :])[None, :].to(
                    self.device
                )
            audio = post_processor.process(
                audio=audio,
                speech_encoder=self.speech_encoder_manager,
                f0=f0,
                c=c,
                vol=vol,
                target_sample=self.target_sample,
                enhancer_adaptive_key=enhancer_adaptive_key,
                hop_length=self.hop_size,
                sid=sid,
                k_step=k_step,
                second_encoding=second_encoding,
            )

            if loudness_envelope_adjustment != 1:
                audio = utils.change_rms(
                    data1=wav,
                    sr1=self.target_sample,
                    data2=audio,
                    sr2=self.target_sample,
                    rate=loudness_envelope_adjustment,
                )
            use_time = time.time() - start
            print("vits use time:{}".format(use_time))
        return audio, audio.shape[-1], n_frames

    def infer_with_f0(
        self,
        speaker: Union[str, torch.Tensor],
        wav: np.ndarray,
        f0: np.ndarray,
        sr: int,
        auto_predict_f0: bool = False,
        noice_scale: float = 0.4,
        frame: int = 0,
        spk_mix: bool = False,
        loudness_envelope_adjustment: float = 1,
        post_processor: PostProcessingManager = None,
        enhancer_adaptive_key: int = 0,
        k_step: int = 100,
        second_encoding: bool = False,
        use_volume: bool = True,
    ) -> Tuple[torch.Tensor, int, int]:
        if (
            not hasattr(self, "audio_resample_transform")
            or self.audio_resample_transform_orig_freq != sr
        ):
            self.audio_resample_transform = partial(
                librosa.resample, orig_sr=sr, target_sr=self.target_sample
            )
            self.audio_resample_transform_orig_freq = sr
        wav = self.audio_resample_transform(wav)
        f0 = torch.from_numpy(f0).to(self.device).to(self.dtype).unsqueeze(0)
        if spk_mix:
            c, f0, uv = self.get_unit_f0(wav, None)
            n_frames = f0.size(1)
            sid = speaker[:, frame : frame + n_frames].transpose(0, 1)
        else:
            speaker_id = self.spk2id.get(speaker)
            if not speaker_id and type(speaker) is int:
                if len(self.spk2id.__dict__) >= speaker:
                    speaker_id = speaker
            if speaker_id is None:
                raise RuntimeError("The name you entered is not in the speaker list!")
            sid = torch.LongTensor([int(speaker_id)]).to(self.device).unsqueeze(0)
            c = self.get_unit(wav, speaker, f0)
            n_frames = f0.size(1)
        c = c.to(self.dtype)
        f0 = f0.to(self.dtype)
        uv = uv.to(self.dtype)
        with torch.no_grad():
            start = time.time()
            vol = (
                self.volume_extractor.extract(
                    torch.FloatTensor(wav).to(self.device)[None, :]
                )[None, :].to(self.device)
                if use_volume
                else None
            )
            audio, f0 = self.net_g_ms.infer(
                c,
                f0=f0,
                g=sid,
                uv=uv,
                predict_f0=auto_predict_f0,
                noice_scale=noice_scale,
                vol=vol,
            )
            audio = audio[0, 0].data.float()
            if self.dtype != torch.float32:
                c = c.to(torch.float32)
                f0 = f0.to(torch.float32)
                uv = uv.to(torch.float32)

            if not use_volume:
                vol = self.volume_extractor.extract(audio[None, :])[None, :].to(
                    self.device
                )
            audio = post_processor.process(
                audio=audio,
                speech_encoder=self.speech_encoder_manager,
                f0=f0,
                c=c,
                vol=vol,
                target_sample=self.target_sample,
                enhancer_adaptive_key=enhancer_adaptive_key,
                hop_length=self.hop_size,
                sid=sid,
                k_step=k_step,
                second_encoding=second_encoding,
            )

            if loudness_envelope_adjustment != 1:
                audio = utils.change_rms(
                    data1=wav,
                    sr1=self.target_sample,
                    data2=audio,
                    sr2=self.target_sample,
                    rate=loudness_envelope_adjustment,
                )
            use_time = time.time() - start
            print("vits use time:{}".format(use_time))
        return audio, audio.shape[-1], n_frames

    def clear_empty(self) -> None:
        # clean up vram
        torch.cuda.empty_cache()

    def unload_model(self) -> None:
        # unload model
        self.net_g_ms = self.net_g_ms.to("cpu")
        del self.net_g_ms
        gc.collect()

    def slice_inference(
        self,
        raw_audio_path: str,
        spk: Union[str, torch.Tensor],
        slice_db: float,
        auto_predict_f0: bool,
        noice_scale: float,
        use_spk_mix: bool = False,
        loudness_envelope_adjustment: float = 1,
        enhancer_adaptive_key: int = 0,
        k_step: int = 100,
        second_encoding: bool = False,
        use_volume: bool = False,
    ) -> np.ndarray:
        if use_spk_mix:
            if len(self.spk2id) == 1:
                spk = self.spk2id.keys()[0]
                use_spk_mix = False
        wav_path = Path(raw_audio_path).with_suffix(".wav")
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks, self.target_sample)
        per_size = int(self.clip_seconds * audio_sr)
        lg_size = int(self.lg_num * audio_sr)
        lg_size_r = int(lg_size * self.lgr_num)
        lg_size_c_l = (lg_size - lg_size_r) // 2
        lg_size_c_r = lg_size - lg_size_r - lg_size_c_l
        lg = np.linspace(0, 1, lg_size_r) if lg_size != 0 else 0

        if use_spk_mix:
            assert len(self.spk2id) == len(spk)
            audio_length = 0
            for slice_tag, data in audio_data:
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
                    if mix[3] < 0.0 or mix[2] < 0.0:
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
                    if step == 0.0:
                        spk_mix_data = torch.zeros(length).to(self.device) + mix[2]
                    else:
                        spk_mix_data = torch.arange(mix[2], mix[3], step).to(
                            self.device
                        )
                    if len(spk_mix_data) < length:
                        num_pad = length - len(spk_mix_data)
                        spk_mix_data = torch.nn.functional.pad(
                            spk_mix_data, [0, num_pad], mode="reflect"
                        ).to(self.device)
                    spk_mix_tensor[i][begin:end] = spk_mix_data[:length]

            spk_mix_ten = torch.sum(spk_mix_tensor, dim=0).unsqueeze(0).to(self.device)
            # spk_mix_tensor[0][spk_mix_ten<0.001] = 1.0
            for i, x in enumerate(spk_mix_ten[0]):
                if x == 0.0:
                    spk_mix_ten[0][i] = 1.0
                    spk_mix_tensor[:, i] = 1.0 / len(spk)
            spk_mix_tensor = spk_mix_tensor / spk_mix_ten
            if not ((torch.sum(spk_mix_tensor, dim=0) - 1.0) < 0.0001).all():
                raise RuntimeError("sum(spk_mix_tensor) not equal 1")
            spk = spk_mix_tensor

        global_frame = 0
        audio = np.array([], dtype=np.float32)
        for slice_tag, data in audio_data:
            print(f"#=====segment start, {round(len(data) / audio_sr, 3)}s======")
            # padd
            length = int(np.ceil(len(data) / audio_sr * self.target_sample))
            if slice_tag:
                print("jump empty segment")
                _audio = np.zeros(length, dtype=np.float32)
                audio = np.concatenate([audio, _audio])
                global_frame += length // self.hop_size
                continue
            if per_size != 0:
                datas = split_list_by_n(data, per_size, lg_size)
            else:
                datas = [data]
            for k, dat in enumerate(datas):
                per_length = (
                    int(np.ceil(len(dat) / audio_sr * self.target_sample))
                    if self.clip_seconds != 0
                    else length
                )
                if self.clip_seconds != 0:
                    print(
                        f"###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======"
                    )
                # padd
                pad_len = int(audio_sr * self.pad_seconds)
                dat = np.concatenate(
                    [
                        np.zeros([pad_len], dtype=np.float32),
                        dat,
                        np.zeros([pad_len], dtype=np.float32),
                    ]
                )
                out_audio, out_sr, out_frame = self.infer(
                    spk,
                    dat,
                    audio_sr,
                    auto_predict_f0=auto_predict_f0,
                    noice_scale=noice_scale,
                    frame=global_frame,
                    spk_mix=use_spk_mix,
                    loudness_envelope_adjustment=loudness_envelope_adjustment,
                    post_processor=self.post_processor_manager,
                    enhancer_adaptive_key=enhancer_adaptive_key,
                    k_step=k_step,
                    second_encoding=second_encoding,
                    use_volume=use_volume,
                )
                global_frame += out_frame
                _audio = out_audio.cpu().numpy()
                pad_len = int(self.target_sample * self.pad_seconds)
                _audio = _audio[pad_len:-pad_len]
                _audio = pad_array(_audio, per_length)
                if lg_size != 0 and k != 0:
                    lg1 = (
                        audio[-(lg_size_r + lg_size_c_r) : -lg_size_c_r]
                        if self.lgr_num != 1
                        else audio[-lg_size:]
                    )
                    lg2 = (
                        _audio[lg_size_c_l : lg_size_c_l + lg_size_r]
                        if self.lgr_num != 1
                        else _audio[0:lg_size]
                    )
                    lg_pre = lg1 * (1 - lg) + lg2 * lg
                    audio = (
                        audio[0 : -(lg_size_r + lg_size_c_r)]
                        if self.lgr_num != 1
                        else audio[0:-lg_size]
                    )
                    audio = np.concatenate([audio, lg_pre])
                    _audio = (
                        _audio[lg_size_c_l + lg_size_r :]
                        if self.lgr_num != 1
                        else _audio[lg_size:]
                    )
                audio = np.concatenate([audio, _audio])
        return audio

    def slice_f0_inference(
        self,
        raw_audio_path: str,
        f0: np.ndarray,
        spk: Union[str, torch.Tensor],
        slice_db: float,
        auto_predict_f0: bool,
        noice_scale: float,
        use_spk_mix: bool = False,
        loudness_envelope_adjustment: float = 1,
        enhancer_adaptive_key: int = 0,
        k_step: int = 100,
        second_encoding: bool = False,
        use_volume: bool = False,
    ):
        if use_spk_mix:
            if len(self.spk2id) == 1:
                spk = self.spk2id.keys()[0]
                use_spk_mix = False
        f0_frame_duration = self.hop_size / self.target_sample  # 0.0116s
        wav_path = Path(raw_audio_path).with_suffix(".wav")
        chunks = slicer.cut(wav_path, db_thresh=slice_db, sr=self.target_sample, hop_size=int(f0_frame_duration*1000))
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks, self.target_sample)
        f0_data = slicer.chunks2f0(f0, chunks, f0_frame_duration)
        per_size = int(self.clip_seconds * audio_sr)
        per_f0_size = int(self.clip_seconds / f0_frame_duration)
        lg_size = int(self.lg_num * audio_sr)
        lg_f0_size = int(self.lg_num / f0_frame_duration)
        lg_size_r = int(lg_size * self.lgr_num)
        lg_size_c_l = (lg_size - lg_size_r) // 2
        lg_size_c_r = lg_size - lg_size_r - lg_size_c_l
        lg = np.linspace(0, 1, lg_size_r) if lg_size != 0 else 0

        if use_spk_mix:
            assert len(self.spk2id) == len(spk)
            audio_length = 0
            for slice_tag, data in audio_data:
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
                    if mix[3] < 0.0 or mix[2] < 0.0:
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
                    if step == 0.0:
                        spk_mix_data = torch.zeros(length).to(self.device) + mix[2]
                    else:
                        spk_mix_data = torch.arange(mix[2], mix[3], step).to(
                            self.device
                        )
                    if len(spk_mix_data) < length:
                        num_pad = length - len(spk_mix_data)
                        spk_mix_data = torch.nn.functional.pad(
                            spk_mix_data, [0, num_pad], mode="reflect"
                        ).to(self.device)
                    spk_mix_tensor[i][begin:end] = spk_mix_data[:length]

            spk_mix_ten = torch.sum(spk_mix_tensor, dim=0).unsqueeze(0).to(self.device)
            # spk_mix_tensor[0][spk_mix_ten<0.001] = 1.0
            for i, x in enumerate(spk_mix_ten[0]):
                if x == 0.0:
                    spk_mix_ten[0][i] = 1.0
                    spk_mix_tensor[:, i] = 1.0 / len(spk)
            spk_mix_tensor = spk_mix_tensor / spk_mix_ten
            if not ((torch.sum(spk_mix_tensor, dim=0) - 1.0) < 0.0001).all():
                raise RuntimeError("sum(spk_mix_tensor) not equal 1")
            spk = spk_mix_tensor

        global_frame = 0
        audio = np.array([], dtype=np.float32)
        for (slice_tag, data), (slice_tag2, data2) in zip(audio_data, f0_data):
            print(f"#=====segment start, {round(len(data) / audio_sr, 3)}s======")
            # padd
            length = int(np.ceil(len(data) / audio_sr * self.target_sample))
            if slice_tag:
                print("jump empty segment")
                _audio = np.zeros(length, dtype=np.float32)
                audio = np.concatenate([audio, _audio])
                global_frame += length // self.hop_size
                continue
            if per_size != 0:
                datas = split_list_by_n(data, per_size, lg_size)
                f0_datas = split_list_by_n(data2, per_f0_size, lg_f0_size)
            else:
                datas = [data]
                f0_datas = [data2]
            for k, (dat, dat2) in enumerate(zip(datas, f0_datas)):
                per_length = (
                    int(np.ceil(len(dat) / audio_sr * self.target_sample))
                    if self.clip_seconds != 0
                    else length
                )   # target length
                if self.clip_seconds != 0:
                    print(
                        f"###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======"
                    )
                # padd
                pad_len = int(audio_sr * self.pad_seconds)
                pad_f0_len = int(self.pad_seconds / f0_frame_duration)
                dat = np.concatenate(
                    [
                        np.zeros([pad_len], dtype=np.float32),
                        dat,
                        np.zeros([pad_len], dtype=np.float32),
                    ]
                )
                dat2 = np.concatenate(
                    [
                        np.zeros([pad_f0_len], dtype=np.float32),
                        dat2,
                        np.zeros([pad_f0_len], dtype=np.float32),
                    ]
                )
                out_audio, out_sr, out_frame = self.infer_with_f0(
                    spk,
                    dat,
                    dat2,
                    audio_sr,
                    auto_predict_f0=auto_predict_f0,
                    noice_scale=noice_scale,
                    frame=global_frame,
                    spk_mix=use_spk_mix,
                    loudness_envelope_adjustment=loudness_envelope_adjustment,
                    post_processor=self.post_processor_manager,
                    enhancer_adaptive_key=enhancer_adaptive_key,
                    k_step=k_step,
                    second_encoding=second_encoding,
                    use_volume=use_volume,
                )
                global_frame += out_frame
                _audio = out_audio.cpu().numpy()
                pad_len = int(self.target_sample * self.pad_seconds)
                _audio = _audio[pad_len:-pad_len]
                _audio = pad_array(_audio, per_length)
                if lg_size != 0 and k != 0:
                    lg1 = (
                        audio[-(lg_size_r + lg_size_c_r) : -lg_size_c_r]
                        if self.lgr_num != 1
                        else audio[-lg_size:]
                    )
                    lg2 = (
                        _audio[lg_size_c_l : lg_size_c_l + lg_size_r]
                        if self.lgr_num != 1
                        else _audio[0:lg_size]
                    )
                    lg_pre = lg1 * (1 - lg) + lg2 * lg
                    audio = (
                        audio[0 : -(lg_size_r + lg_size_c_r)]
                        if self.lgr_num != 1
                        else audio[0:-lg_size]
                    )
                    audio = np.concatenate([audio, lg_pre])
                    _audio = (
                        _audio[lg_size_c_l + lg_size_r :]
                        if self.lgr_num != 1
                        else _audio[lg_size:]
                    )
                audio = np.concatenate([audio, _audio])
        return audio

    def realtime_infer(
        self,
        speaker_id: Union[str, int],
        f_pitch_change: float,
        input_wav_path: str,
        cluster_infer_ratio: float = 0,
        auto_predict_f0: bool = False,
        noice_scale: float = 0.4,
        f0_filter: bool = False,
    ) -> np.ndarray:
        import maad

        audio, sr = torchaudio.load(input_wav_path)
        audio = audio.cpu().numpy()[0]
        temp_wav = io.BytesIO()
        if self.last_chunk is None:
            input_wav_path.seek(0)

            audio, sr, _ = self.infer(
                speaker_id,
                f_pitch_change,
                input_wav_path,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noice_scale=noice_scale,
                f0_filter=f0_filter,
            )

            audio = audio.cpu().numpy()
            self.last_chunk = audio[-self.pre_len :]
            self.last_o = audio
            return audio[-self.chunk_len :]
        else:
            audio = np.concatenate([self.last_chunk, audio])
            soundfile.write(temp_wav, audio, sr, format="wav")
            temp_wav.seek(0)

            audio, sr, _ = self.infer(
                speaker_id,
                f_pitch_change,
                temp_wav,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noice_scale=noice_scale,
                f0_filter=f0_filter,
            )

            audio = audio.cpu().numpy()
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            self.last_chunk = audio[-self.pre_len :]
            self.last_o = audio
            return ret[self.chunk_len : 2 * self.chunk_len]


class SVCManager:
    def __init__(self):
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self.svc_modes: List[str] = ["so-vits-svc-4.0"]
        self.svc: Optional[str] = None
        self.svc_object: Optional[Svc] = None
        self.hps_ms: Optional[Union[HParams, InferHParams]] = None
        self.model_path: Optional[str] = None
        self.use_spk_mix: bool = False

    def initialize(
        self,
        svc: str,
        model_path: str,
        config: Union[HParams, InferHParams],
        threshold: float,
        use_spk_mix: bool,
        pad_seconds: float = 0.5,
        clip_seconds: float = 0,
        lg_num: float = 0,
        lgr_num: float = 0.75,
        device: str = "cpu",
    ):
        # if there is no svc model, or svc, model_path, config or device changes, reload the model
        if (
            self.svc != svc
            or self.svc_object is None
            or self.hps_ms != config
            or self.device != device
            or self.model_path != model_path
            or self.use_spk_mix != use_spk_mix
        ):
            self.device = device
            self.model_path = model_path
            self.use_spk_mix = use_spk_mix

            self.pad_seconds = pad_seconds
            self.clip_seconds = clip_seconds
            self.lg_num = lg_num
            self.lgr_num = lgr_num
            self.threshold = threshold

            self.svc = svc
            self.svc_object = self.get_svc(
                svc,
                model_path,
                config,
                threshold,
                use_spk_mix,
                pad_seconds,
                clip_seconds,
                lg_num,
                lgr_num,
                device,
            )
            self.hps_ms = config
            self.spk2id = self.svc_object.spk2id
            self.speech_encoder = self.svc_object.speech_encoder
            self.dtype = self.svc_object.dtype
        else:
            # else you can just use the existing model. Just change some other parameters
            self.threshold = threshold
            self.pad_seconds = pad_seconds
            self.clip_seconds = clip_seconds
            self.lg_num = lg_num
            self.lgr_num = lgr_num
            self.svc_object.update(
                pad_seconds, clip_seconds, lg_num, lgr_num, threshold
            )

    def get_svc(
        self,
        svc: str,
        model_path: str,
        config: Union[HParams, InferHParams],
        threshold: float,
        use_spk_mix: bool,
        pad_seconds: float,
        clip_seconds: float,
        lg_num: float,
        lgr_num: float,
        device: str,
    ) -> Svc:
        if svc == "so-vits-svc-4.0":
            return Svc(
                model_path,
                config,
                threshold,
                use_spk_mix,
                pad_seconds,
                clip_seconds,
                lg_num,
                lgr_num,
                device,
            )
        else:
            raise Exception(f"Unsupported svc: {svc}, available: {self.svc_modes}")

    def load_components(
        self,
        f0: f0Manager,
        speech_encoder: SpeechEncoderManager,
        post_processing: PostProcessingManager,
    ):
        self.svc_object.initialize(speech_encoder, f0, post_processing)

    def f0_to_wav(self, f0: torch.Tensor) -> Tuple[np.ndarray, int]:
        # [1, T] -> [T]
        f0 = f0.squeeze(0)
        target_sr = self.svc_object.target_sample
        hop_size = self.svc_object.hop_size
        clip_seconds = 5.0
        pad_seconds = 0.5

        # f0의 길이를 기반으로 전체 오디오 길이 계산
        total_frames = len(f0)
        total_seconds = total_frames * hop_size / target_sr

        # 청크 크기 및 linear gradient 크기 계산
        chunk_size = int(clip_seconds * target_sr / hop_size)
        chunk_sample = int(clip_seconds * target_sr)
        pad_size = int(pad_seconds * target_sr / hop_size)
        pad_sample = int(pad_seconds * target_sr)

        # 결과 저장을 위한 리스트
        wav = np.array([])

        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)

            f0_chunk = f0[start:end]

            # Add padding
            f0_chunk = torch.nn.functional.pad(
                f0_chunk, (pad_size, pad_size), mode="constant", value=0
            )

            f0_tensor = f0_chunk.unsqueeze(0).to(self.svc_object.device)  # [1, T]

            # apply pitch2wav
            wav_chunk = self.svc_object.net_g_ms.pitch2wav(f0_tensor)  # [S] numpy

            wav_chunk = wav_chunk[pad_sample:-pad_sample]  # remove padding

            wav = np.concatenate([wav, wav_chunk])

        # Return the concatenated wav chunks numpy
        return wav, target_sr

    def get_f0(self, raw_audio_path: str) -> Tuple[torch.Tensor, int, int]:
        # Load the audio file to target sampling rate
        wav, _, sr, hop_size = self.svc_object.load_wav(
            raw_audio_path
        )  # sr is target length.
        f0, _ = self.svc_object.f0_manager.compute_f0_uv_tran(
            wav
        )  # [1, T], T is seconds * sr / hop_length

        return f0, sr, hop_size

    def infer_slice(
        self,
        raw_audio_path: str,
        spk: Union[str, torch.Tensor],
        slice_db: float,
        auto_predict_f0: bool,
        noice_scale: float,
        use_spk_mix: bool,
        loudness_envelope_adjustment: float,
        enhancer_adaptive_key: int,
        k_step: int,
        second_encoding: bool,
        use_volume: bool,
    ) -> np.ndarray:
        audio = self.svc_object.slice_inference(
            raw_audio_path,
            spk,
            slice_db,
            auto_predict_f0,
            noice_scale,
            use_spk_mix,
            loudness_envelope_adjustment,
            enhancer_adaptive_key,
            k_step,
            second_encoding,
            use_volume,
        )
        return audio

    def infer_slice_with_f0(
        self,
        raw_audio_path: str,
        f0: np.ndarray,
        spk: Union[str, torch.Tensor],
        slice_db: float,
        auto_predict_f0: bool,
        noice_scale: float,
        use_spk_mix: bool,
        loudness_envelope_adjustment: float,
        enhancer_adaptive_key: int,
        k_step: int,
        second_encoding: bool,
        use_volume: bool,
    ):
        f0 = f0.squeeze() # [T]
        audio = self.svc_object.slice_f0_inference(
            raw_audio_path,
            f0,
            spk,
            slice_db,
            auto_predict_f0,
            noice_scale,
            use_spk_mix,
            loudness_envelope_adjustment,
            enhancer_adaptive_key,
            k_step,
            second_encoding,
            use_volume,
        )
        return audio

    def realtime_infer(self):
        pass

    def unload_model(self):
        self.svc_object.unload_model()
