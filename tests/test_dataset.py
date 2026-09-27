"""AudioDataset on a table mixing segment-indexed and one-row-per-file corpora.

Concatenating a segment index (``start_s``/``end_s`` per row) with a corpus that
has one row per file leaves those columns present but NaN on the file rows.  The
whole-file path has to stay reachable for them, which is what these tests pin.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from audio_utils.data.dataset import AudioDataset

SAMPLE_RATE = 16_000


@pytest.fixture
def wav_file(tmp_path):
    """A 4 s ramp, so any window's content identifies where it was read from."""
    path = tmp_path / "ramp.wav"
    n = 4 * SAMPLE_RATE
    sf.write(path, np.linspace(0.0, 1.0, n, dtype="float32"), SAMPLE_RATE)
    return str(path)


@pytest.fixture
def stereo_file(tmp_path):
    """Channel 0 is a sine, channel 1 is silence — a stand-in for AVID's EGG."""
    path = tmp_path / "stereo.wav"
    t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    left = np.sin(2 * math.pi * 200.0 * t).astype("float32")
    sf.write(path, np.stack([left, np.zeros_like(left)], axis=1), SAMPLE_RATE)
    return str(path)


def test_nan_bounds_read_the_whole_file(wav_file):
    """A NaN-bounded row must span its file, not inherit the segment path."""
    metadata = pd.DataFrame({
        "signal_path": [wav_file, wav_file],
        "start_s": [1.0, float("nan")],
        "end_s": [2.0, float("nan")],
    })
    dataset = AudioDataset(metadata, target_sr=SAMPLE_RATE, chunk_s=None)

    assert dataset.ends[1] is None, "NaN end_s must become the whole-file sentinel"
    assert dataset[1]["wav"].shape[-1] == 4 * SAMPLE_RATE
    assert dataset[0]["wav"].shape[-1] == SAMPLE_RATE


def test_bounded_row_reads_its_own_region(wav_file):
    """The segment path still works, and reads the region it was given."""
    metadata = pd.DataFrame({
        "signal_path": [wav_file],
        "start_s": [2.0],
        "end_s": [3.0],
    })
    wav = AudioDataset(metadata, target_sr=SAMPLE_RATE, chunk_s=None)[0]["wav"]
    # The ramp runs 0..1 over 4 s, so the window starting at 2 s starts at 0.5.
    assert float(wav[0, 0]) == pytest.approx(0.5, abs=1e-3)


def test_nan_bounds_survive_the_chunked_path(wav_file):
    """``chunk_s`` takes a different branch, whose NaN guards all compare False."""
    metadata = pd.DataFrame({
        "signal_path": [wav_file, wav_file],
        "start_s": [1.0, float("nan")],
        "end_s": [2.0, float("nan")],
    })
    dataset = AudioDataset(metadata, target_sr=SAMPLE_RATE, chunk_s=0.5, train=False)
    for index in range(2):
        assert dataset[index]["wav"].shape[-1] == SAMPLE_RATE // 2


def test_absent_bound_columns_are_unchanged(wav_file):
    """A table with no bounds at all keeps behaving as it always did."""
    metadata = pd.DataFrame({"signal_path": [wav_file]})
    dataset = AudioDataset(metadata, target_sr=SAMPLE_RATE, chunk_s=None)
    assert dataset.starts == [0.0] and dataset.ends == [None]
    assert dataset[0]["wav"].shape[-1] == 4 * SAMPLE_RATE


def test_nan_channel_mono_mixes_and_an_index_selects(stereo_file):
    """The channel guard already existed; pinned here beside its sibling."""
    metadata = pd.DataFrame({
        "signal_path": [stereo_file, stereo_file],
        "channel": [0, float("nan")],
    })
    dataset = AudioDataset(metadata, target_sr=SAMPLE_RATE, chunk_s=None)
    assert dataset.channels == [0, -1]

    selected = dataset[0]["wav"]          # channel 0 alone
    mixed = dataset[1]["wav"]             # averaged with the silent channel
    assert torch.allclose(mixed, selected / 2, atol=1e-6)
