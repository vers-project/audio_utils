"""Collate functions for DataLoader."""
from __future__ import annotations

from typing import Callable

from audio_utils.data.audio_batch import AudioBatch


def make_collate_fn(sample_rate: int) -> Callable:
    """Return a collate function that packs waveforms into an AudioBatch.

    Expects each item in the batch to be a dict with a ``"wav"`` key holding
    a (1, T) tensor.  Produces a dict with ``"wav": AudioBatch``.

    Parameters
    ----------
    sample_rate : sample rate of the waveforms (stored in the AudioBatch).
    """

    def collate_fn(batch: list[dict]) -> dict:
        wavs = [item["wav"] for item in batch]
        return {"wav": AudioBatch.from_list(wavs, sample_rate)}

    return collate_fn
