"""Stateless audio transform functions.

All operations take tensors and return tensors — no side effects.
Use these directly in Dataset.__getitem__ or as pre-processing steps.

Chunking convention
-------------------
Chunk boundaries are expressed in **seconds**, and a signal shorter than the
requested chunk is returned **unchanged, not zero-padded**.  Padding is the
collate function's job: ``AudioBatch.from_list`` derives ``lengths`` from the
tensors it receives, so a Dataset that pads has already destroyed the
information that those samples are fabricated — the resulting mask declares
them valid audio and every masked loss then averages over invented silence.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import Tensor

# Read slightly more than the requested chunk so resampling to another rate
# cannot leave it a sample short; callers trim to the exact length.
_CHUNK_SLACK_S = 0.01

# Level reported for a digitally silent signal, in place of -inf.
SILENCE_DBFS = -120.0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def audio_info(path: str | Path) -> tuple[float, int]:
    """Duration in seconds and sample rate of an audio file, without decoding it."""
    info = sf.info(str(path))
    return info.frames / info.samplerate, info.samplerate


def load_audio(
    path: str | Path,
    offset_s: float = 0.0,
    duration_s: float | None = None,
) -> tuple[Tensor, int]:
    """Load audio from path.  Returns (wav, sample_rate) where wav is (C, T).

    ``offset_s`` and ``duration_s`` read a single window of the file: soundfile
    seeks to the offset instead of decoding what precedes it, so the cost scales
    with the window rather than with the file.  This matters for long
    recordings — reading 1 s out of a 10-minute FLAC is ~500x faster than
    decoding it whole.  ``duration_s=None`` reads to the end, and reading past
    the end returns what is available.
    """
    info = sf.info(str(path))
    start = round(offset_s * info.samplerate)
    frames = -1 if duration_s is None else round(duration_s * info.samplerate)
    data, sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), sr


def load_audio_chunk(
    path: str | Path,
    chunk_s: float,
    start_s: float = 0.0,
    end_s: float | None = None,
    random_offset: bool = False,
) -> tuple[Tensor, int]:
    """Read one ``chunk_s`` window taken from within ``[start_s, end_s]``.

    Only that window is decoded, so the cost is independent of the file
    duration.

    Parameters
    ----------
    start_s, end_s : bound the region the window is drawn from.  ``end_s=None``
                     means the end of the file; passing it explicitly (a VAD
                     segment's bounds, say) also skips the header read that
                     would otherwise be needed to find the duration.
    random_offset  : draw the window uniformly from the region.  The default
                     starts it at ``start_s``, so a call returns the same audio
                     every time unless randomness is asked for by name.

    A region shorter than ``chunk_s`` is returned whole, never padded.
    """
    if end_s is None:
        end_s, _ = audio_info(path)
    available_s = end_s - start_s
    if available_s <= 0:
        raise ValueError(f"Empty region [{start_s}, {end_s}) requested from {path}")
    if available_s <= chunk_s:
        return load_audio(path, start_s, available_s)
    offset_s = random.uniform(start_s, end_s - chunk_s) if random_offset else start_s
    return load_audio(path, offset_s, chunk_s + _CHUNK_SLACK_S)


def resample(wav: Tensor, from_sr: int, to_sr: int) -> Tensor:
    """Resample wav from from_sr to to_sr.  No-op if rates match."""
    if from_sr == to_sr:
        return wav
    return AF.resample(wav, from_sr, to_sr)


# ---------------------------------------------------------------------------
# Channel handling
# ---------------------------------------------------------------------------


def mono_mix(wav: Tensor) -> Tensor:
    """Average all channels to mono, preserving the channel dim: (1, T)."""
    if wav.shape[0] == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def select_channel(wav: Tensor, channel: int) -> Tensor:
    """Extract one channel, preserving the channel dim: (1, T).

    Parameters
    ----------
    wav     : (C, T) audio tensor.
    channel : 0-based index of the channel to keep.  Pass -1 to fall back
              to ``mono_mix`` (average all channels).
    """
    if channel < 0:
        return mono_mix(wav)
    return wav[channel : channel + 1]


# ---------------------------------------------------------------------------
# Amplitude
# ---------------------------------------------------------------------------


def peak_normalize(wav: Tensor, target_peak: float = 0.9) -> Tensor:
    """Scale wav so its peak absolute value equals target_peak.

    Silent signals (peak == 0) are returned unchanged.
    """
    peak = wav.abs().max()
    if peak > 0:
        wav = wav * (target_peak / peak)
    return wav


def rms(wav: Tensor) -> Tensor:
    """Root-mean-square amplitude over all channels and samples."""
    return wav.float().pow(2).mean().sqrt()


def rms_dbfs(wav: Tensor) -> float:
    """RMS level in dBFS, relative to a full-scale amplitude of 1.0.

    A full-scale sine therefore reads ≈ -3.01 dBFS, not 0.  Digital silence
    returns ``SILENCE_DBFS`` rather than -inf, so that a silent signal yields a
    finite, sortable, obviously-wrong number instead of poisoning every mean
    computed over it.
    """
    value = float(rms(wav))
    return SILENCE_DBFS if value <= 0.0 else 20.0 * math.log10(value)


def peak_dbfs(wav: Tensor) -> float:
    """Peak level in dBFS.  With ``rms_dbfs`` this gives the crest factor, which
    is what flags a clipped or a heavily compressed signal."""
    value = float(wav.abs().max())
    return SILENCE_DBFS if value <= 0.0 else 20.0 * math.log10(value)


def rms_normalize(wav: Tensor, target_dbfs: float = -27.0) -> Tensor:
    """Scale wav so its RMS level equals target_dbfs.

    Prefer this over ``peak_normalize`` whenever the point is to remove level as
    a variable: peak level is set by a single sample and says very little about
    how loud a signal is.

    Silent signals are returned unchanged — no gain gives silence a level.  The
    result is **not** peak-limited, deliberately: clipping here would distort the
    very signal under measurement, so it is better to pick a target with
    headroom.  The -27 dBFS default leaves room for a 27 dB crest factor,
    comfortably above conversational speech.
    """
    current = rms(wav)
    if current <= 0.0:
        return wav
    return wav * (10.0 ** (target_dbfs / 20.0) / current)


# ---------------------------------------------------------------------------
# Chunking / padding
# ---------------------------------------------------------------------------


def random_chunk(wav: Tensor, sr: int, duration_s: float) -> Tensor:
    """Randomly crop wav to duration_s seconds.  Returns it unchanged if shorter.

    Prefer ``load_audio_chunk`` when cropping a file straight from disk: this
    one requires the whole signal to be decoded first.
    """
    chunk_len = int(duration_s * sr)
    T = wav.shape[-1]
    if T <= chunk_len:
        return wav
    start = random.randint(0, T - chunk_len)
    return wav[..., start : start + chunk_len]


def fixed_chunk(wav: Tensor, sr: int, duration_s: float, start_s: float = 0.0) -> Tensor:
    """Deterministically crop wav to duration_s seconds starting at start_s.

    A slice running past the end of the signal is returned short, not padded.
    """
    start = int(start_s * sr)
    end = start + int(duration_s * sr)
    return wav[..., start:end]


def pad_to_multiple(wav: Tensor, multiple: int) -> Tensor:
    """Right-pad wav with zeros so its length is a multiple of ``multiple``."""
    T = wav.shape[-1]
    remainder = T % multiple
    if remainder == 0:
        return wav
    return F.pad(wav, (0, multiple - remainder))
