import glob
import json
import os.path

import torch
import sounddevice as sd
import soundfile as sf
from PyQt5.QtWidgets import QDesktopWidget, QHBoxLayout, QVBoxLayout


def get_available_devices():
    devices = []

    # Check for CPU
    devices.append(("CPU", "cpu"))

    # Check for CUDA devices
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        for i in range(cuda_count):
            cuda_name = torch.cuda.get_device_name(i)
            devices.append((f"{cuda_name}", f"cuda:{i}"))

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append(("Apple MPS", "mps"))

    # # Check for xla
    # try:
    #     import torch_xla.core.xla_model as xm
    #     if xm.xrt_world_size() > 0:
    #         devices.append(("TPU", "xla"))
    # except ImportError:
    #     pass

    return devices


def get_audio_devices():
    devices = sd.query_devices()
    input_devices = []
    output_devices = []

    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((device['name'], i))
        if device['max_output_channels'] > 0:
            output_devices.append((device['name'], i))

    return input_devices, output_devices

def calculateWindowSize():
    screen = QDesktopWidget().availableGeometry()

    # set 2/3 of the screen size as the default size
    target_width = int(screen.width() * 2 / 3)
    target_height = int(screen.height() * 2 / 3)

    # min size (y: 800, x: 1200)
    width = max(target_width, 1200)
    height = max(target_height, 800)

    # Locate the window in the center of the screen
    left = int((screen.width() - width) / 2)
    top = int((screen.height() - height) / 2)

    return (left, top, width, height)

def save_json_file(folder_path, file_path, data):
    with open(f"{folder_path}/{file_path}.json", 'w+', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_json_file(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # try to load all json files in the folder_path with glob
    jsons = glob.glob(f"{folder_path}/*.json")
    presets = {}
    for json_file in jsons:
        preset_name = os.path.splitext(os.path.basename(json_file))[0]
        print(f"Loading {preset_name}")
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            presets[preset_name] = data

    return presets

def create_QHBox(widgets, parent=None):
    hbox = QHBoxLayout(parent)
    for widget in widgets:
        hbox.addWidget(widget)
    return hbox

def create_QVBox(widgets, parent=None):
    vbox = QVBoxLayout(parent)
    for widget in widgets:
        vbox.addWidget(widget)
    return vbox


def get_supported_file_types_concat() -> str:
    # "Audio Files (*.wav *.mp3 *.flac)"
    formats = [f"*.{ext.lower()}" for ext in sf.available_formats().keys()]
    return "Audio Files" + " ".join(formats)


# def split_silence(
#     audio: ndarray[Any, dtype[float32]],
#     top_db: int = 40,
#     ref: float | Callable[[ndarray[Any, dtype[float32]]], float] = 1,
#     frame_length: int = 2048,
#     hop_length: int = 512,
#     aggregate: Callable[[ndarray[Any, dtype[float32]]], float] = np.mean,
#     max_chunk_length: int = 0,
# ) -> Iterable[Chunk]:
#     non_silence_indices = librosa.effects.split(
#         audio,
#         top_db=top_db,
#         ref=ref,
#         frame_length=frame_length,
#         hop_length=hop_length,
#         aggregate=aggregate,
#     )
#     last_end = 0
#     for start, end in non_silence_indices:
#         if start != last_end:
#             yield Chunk(
#                 is_speech=False, audio=audio[last_end:start], start=last_end, end=start
#             )
#         while max_chunk_length > 0 and end - start > max_chunk_length:
#             yield Chunk(
#                 is_speech=True,
#                 audio=audio[start : start + max_chunk_length],
#                 start=start,
#                 end=start + max_chunk_length,
#             )
#             start += max_chunk_length
#         if end - start > 0:
#             yield Chunk(is_speech=True, audio=audio[start:end], start=start, end=end)
#         last_end = end
#     if last_end != len(audio):
#         yield Chunk(
#             is_speech=False, audio=audio[last_end:], start=last_end, end=len(audio)
#         )
