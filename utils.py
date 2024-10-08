import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
import traceback
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union, Any, Optional, Tuple, Dict, List

import librosa
import numpy as np
import soundfile
import faiss
import torch
from scipy.io.wavfile import read
from sklearn.cluster import MiniBatchKMeans
from torch.nn import functional as F

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.WARN)
logger = logging

# TODO
f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def format_wav(audio_path: Union[str, Path]):
    if Path(audio_path).suffix == ".wav":
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def change_rms(
    data1: np.ndarray, sr1: int, data2: torch.Tensor, sr2: int, rate: float
) -> torch.Tensor:
    # data1 is original audio and data2 is converted audio
    # rate 0 -> data1, rate 1 -> data2
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(
        y=data2.detach().cpu().numpy(), frame_length=sr2 // 2 * 2, hop_length=sr2 // 2
    )
    rms1 = torch.from_numpy(rms1).to(data2.device)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2).to(data2.device)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(
        rms2, torch.tensor(rate - 1)
    )
    return data2


def train_index(
    spk_name: str, root_dir: str = "dataset/44k/"
) -> (
    faiss.IndexIVF
):  # from: RVC https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    n_cpu = cpu_count()
    print("The feature index is constructing.")
    exp_dir = os.path.join(root_dir, spk_name)
    listdir_res = []
    for file in os.listdir(exp_dir):
        if ".wav.soft.pt" in file:
            listdir_res.append(os.path.join(exp_dir, file))
    if len(listdir_res) == 0:
        raise Exception("You need to run preprocess_hubert_f0.py!")
    npys = []
    for name in sorted(listdir_res):
        phone = torch.load(name)[0].transpose(-1, -2).numpy()
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        # if(1):
        info = "Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0]
        print(info)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception:
            info = traceback.format_exc()
            print(info)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(big_npy.shape[1], "IVF%s,Flat" % n_ivf)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    # faiss.write_index(
    #     index,
    #     f"added_{spk_name}.index"
    # )
    print("Successfully build index")
    return index


class HParams:
    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self) -> List[str]:
        return list(self.__dict__.keys())

    def items(self) -> List[Tuple[str, Any]]:
        return list(self.__dict__.items())

    def values(self) -> List[Any]:
        return list(self.__dict__.values())

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        return setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __repr__(self) -> str:
        return self.__dict__.__repr__()

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.__dict__.get(key, default)


class InferHParams(HParams):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = InferHParams(**v)
            self[k] = v

    def __getattr__(self, index: str) -> Any:
        return self.get(index)


class Volume_Extractor:
    def __init__(self, hop_size: int = 512):
        self.hop_size = hop_size

    def extract(
        self, audio: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:  # audio: 2d tensor array
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).float()
        n_frames = int(audio.size(-1) // self.hop_size)
        audio2 = audio**2
        audio2 = torch.nn.functional.pad(
            audio2,
            (int(self.hop_size // 2), int((self.hop_size + 1) // 2)),
            mode="reflect",
        )
        volume = torch.nn.functional.unfold(
            audio2[:, None, None, :], (1, self.hop_size), stride=self.hop_size
        )[:, :, :n_frames].mean(dim=1)[0]
        volume = torch.sqrt(volume)
        return volume


def f0_to_coarse(f0: torch.Tensor) -> torch.Tensor:
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.0
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    # torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    skip_optimizer: bool = False,
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], float, int]:
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    model = model.to(list(saved_state_dict.values())[0].dtype)
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "dec" in k or "disc" in k
            # print("load", k)
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except Exception:
            if "enc_q" not in k or "emb_g" not in k:
                print(
                    "%s is not in the checkpoint,please check your checkpoint.If you're using pretrain model,just ignore this warning."
                    % k
                )
                logger.info("%s is not in the checkpoint" % k)
                new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("load ")
    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )
    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    iteration: int,
    checkpoint_path: str,
):
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def clean_checkpoints(
    path_to_models: str = "logs/44k/",
    n_ckpts_to_keep: int = 2,
    sort_by_time: bool = True,
):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]

    def name_key(_f):
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))

    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x):
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (x_sorted("G")[:-n_ckpts_to_keep] + x_sorted("D")[:-n_ckpts_to_keep])
    ]

    def del_info(fn):
        return logger.info(f".. Free up space by deleting ckpt {fn}")

    def del_routine(x):
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def summarize(
    writer: Any,
    global_step: int,
    scalars: Dict[str, float] = {},
    histograms: Dict[str, np.ndarray] = {},
    images: Dict[str, np.ndarray] = {},
    audios: Dict[str, np.ndarray] = {},
    audio_sampling_rate: int = 22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path: str, regex: str = "G_*.pth") -> str:
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(
    alignment: np.ndarray, info: Optional[str] = None
) -> np.ndarray:
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path: str) -> Tuple[torch.FloatTensor, int]:
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename: str, split: str = "|") -> List[List[str]]:
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init: bool = True) -> HParams:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/config.json",
        help="JSON file for configuration",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_dir(model_dir: str) -> HParams:
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(
    config_path: str, infer_mode: bool = False
) -> Union[HParams, InferHParams]:
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config) if not infer_mode else InferHParams(**config)
    return hparams


def repeat_expand_2d(
    content: torch.Tensor, target_len: int, mode: str = "left"
) -> torch.Tensor:
    # content : [h, t]
    return (
        repeat_expand_2d_left(content, target_len)
        if mode == "left"
        else repeat_expand_2d_other(content, target_len, mode)
    )


def repeat_expand_2d_left(content: torch.Tensor, target_len: int) -> torch.Tensor:
    # content : [h, t]

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(
        content.device
    )
    temp = torch.arange(src_len + 1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos + 1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target


# mode : 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'
def repeat_expand_2d_other(
    content: torch.Tensor, target_len: int, mode: str = "nearest"
) -> torch.Tensor:
    # content : [h, t]
    content = content[None, :, :]
    target = F.interpolate(content, size=target_len, mode=mode)[0]
    return target


def pad_array(arr: np.ndarray, target_length: int) -> np.ndarray:
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr[:target_length]
    else:
        pad_width = target_length - current_length
        return np.pad(arr, (0, pad_width), "constant", constant_values=0)
