from __future__ import annotations

from typing import Any, Iterable, Callable, Dict, List, Tuple

import attrs
import librosa
import numpy as np
import torch
import torchaudio


class Slicer:
    def __init__(
        self,
        sr: int = 44100,
        threshold: float = -40.0,
        hop_size: int = 512,
        min_silence_seconds: float = 1.0,
        min_chunk_seconds: float = 0.5,
        max_chunk_seconds: float = 40.0,
        absolute_threshold: bool = True,
    ):
        """
        Initialize the slicer.
        :param sr: target audio sample rate
        :param threshold: the threshold in dB below which a frame is considered silent
        :param min_length: the minimum length of a slice in samples
        :param hop_size: the hop size in samples
        """
        self.sr = sr
        self.threshold = -threshold
        self.hop_size = hop_size
        self.min_silence_length = int(min_silence_seconds * sr)
        self.min_chunk_length = int(min_chunk_seconds * sr)
        self.max_chunk_length = int(max_chunk_seconds * sr)
        self.absolute_threshold = absolute_threshold

    def slice(self, audio):
        # Convert stereo to mono if necessary
        if len(audio.shape) > 1:
            samples = librosa.to_mono(audio)
        else:
            samples = audio

        # Return a single chunk if the audio is shorter than the minimum length
        if samples.shape[0] <= self.min_silence_length:
            return {"0": {"slice": False, "split_time": f"0,{len(samples)}"}}

        # Detect non-silent regions
        non_silence_indices = librosa.effects.split(
            samples,
            top_db=self.threshold,
            ref=1.0 if self.absolute_threshold else np.max,
            frame_length=2 * self.hop_size,
            hop_length=self.hop_size,
        )

        last_end = 0
        chunks = {}
        chunk_id = 0
        merge_chunk = False

        for start, end in non_silence_indices:
            # Process silent region if it's long enough
            if start - last_end >= self.min_silence_length or chunk_id == 0:
                chunks[str(chunk_id)] = {
                    "slice": True,
                    "split_time": f"{last_end},{start}"
                }
                chunk_id += 1
                last_end = start
            else:
                merge_chunk = True

            if merge_chunk:
                # scenario1: Audio + Short Silence + Audio ( Merge two audio chunks with short silence in between)
                merge_chunk = False
                chunk_id -= 1
                current_start = int(chunks[str(chunk_id)]["split_time"].split(',')[0])
            else:
                # normal scenario
                current_start = last_end
            while end - current_start > 0:
                chunk_length = min(self.max_chunk_length, end - current_start)
                # Merge with previous chunk if current chunk is too short and previous is non-silent
                # scenario2: Long Silence + Short Audio ( Even if the audio is short, it should be a separate chunk if prev chunk is silent)
                if (chunk_length < self.min_chunk_length and chunk_id > 0 and
                        not chunks[str(chunk_id - 1)]["slice"]):
                    prev_chunk = chunks[str(chunk_id - 1)]
                    prev_start = int(prev_chunk["split_time"].split(',')[0])
                    chunks[str(chunk_id - 1)] = {
                        "slice": False,
                        "split_time": f"{prev_start},{current_start + chunk_length}"
                    }
                else:
                    # Create a new chunk
                    chunks[str(chunk_id)] = {
                        "slice": False,
                        "split_time": f"{current_start},{current_start + chunk_length}"
                    }
                    chunk_id += 1

                current_start += chunk_length

            last_end = end

        chunks[str(chunk_id)] = {
            "slice": True,
            "split_time": f"{last_end},{len(samples)}"
        }

        return chunks


def cut(audio_path, db_thresh=-30, min_len=0.5, sr=None, hop_size=512):
    audio, sr = librosa.load(audio_path, sr=sr)
    slicer = Slicer(sr=sr, threshold=db_thresh, min_chunk_seconds=min_len, hop_size=hop_size)
    chunks = slicer.slice(audio)
    return chunks


def cut_with_audio(audio, db_thresh=-30, min_len=0.5, sr=44100, hop_size=512):
    slicer = Slicer(sr=sr, threshold=db_thresh, min_chunk_seconds=min_len, hop_size=hop_size)
    chunks = slicer.slice(audio)
    return chunks


def chunks2audio(audio_path, chunks, sr=44100, dtype=np.float32):
    chunks = dict(chunks)
    # Load the audio file
    audio, _ = librosa.load(audio_path, sr=sr, mono=False, dtype=dtype)

    # Check if the audio is stereo
    if audio.ndim > 1 and audio.shape[0] == 2:
        # Convert to mono by averaging the channels
        audio = np.mean(audio, axis=0)
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        result.append((v["slice"], audio[int(tag[0]) : int(tag[1])]))
    return result, sr


def chunks2f0(
    f0: np.ndarray, chunks: Dict[str, Any], hop_size: int
) -> List[Tuple[bool, np.ndarray]]:
    """
    Slice the f0 data according to the chunks.

    Args:
        f0 (np.ndarray): The f0 data array.
        chunks (Dict[str, Any]): The chunks dictionary returned by the cut method.

    Returns:
        List[Tuple[bool, np.ndarray]]: A list of tuples, each containing a boolean (is_slice)
                                       and the corresponding f0 chunk.
    """
    result = []
    for k, v in chunks.items():
        start, end = map(int, v["split_time"].split(","))
        start = int(start / hop_size)
        end = int(end / hop_size)
        f0_chunk = f0[start:end]
        result.append((v["slice"], f0_chunk))
    return result
