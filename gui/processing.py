from typing import Callable, Any, Iterable

import attrs
import librosa
import numpy as np
from numpy import ndarray, dtype, float32

@attrs.frozen(kw_only=True)
class Chunk:
    is_speech: bool
    audio: ndarray[Any, dtype[float32]]
    start: int
    end: int

    @property
    def duration(self) -> float32:
        # return self.end - self.start
        return float32(self.audio.shape[0])

    def __repr__(self) -> str:
        return f"Chunk(Speech: {self.is_speech}, {self.duration})"

def split_silence(
    audio: ndarray[Any, dtype[float32]],
    top_db: int = 40,
    ref: float | Callable[[ndarray[Any, dtype[float32]]], float] = 1,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: Callable[[ndarray[Any, dtype[float32]]], float] = np.mean,
    max_chunk_length: int = 0,
) -> Iterable[Chunk]:
    non_silence_indices = librosa.effects.split(
        audio,
        top_db=top_db,
        ref=ref,
        frame_length=frame_length,
        hop_length=hop_length,
        aggregate=aggregate,
    )
    last_end = 0
    for start, end in non_silence_indices:
        if start != last_end:
            yield Chunk(
                is_speech=False, audio=audio[last_end:start], start=last_end, end=start
            )
        while max_chunk_length > 0 and end - start > max_chunk_length:
            yield Chunk(
                is_speech=True,
                audio=audio[start : start + max_chunk_length],
                start=start,
                end=start + max_chunk_length,
            )
            start += max_chunk_length
        if end - start > 0:
            yield Chunk(is_speech=True, audio=audio[start:end], start=start, end=end)
        last_end = end
    if last_end != len(audio):
        yield Chunk(
            is_speech=False, audio=audio[last_end:], start=last_end, end=len(audio)
        )