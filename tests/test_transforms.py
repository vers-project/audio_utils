"""Amplitude transforms: RMS level and RMS normalisation."""
from __future__ import annotations

import math

import pytest
import torch

from audio_utils.data.transforms import (
    SILENCE_DBFS,
    peak_dbfs,
    peak_normalize,
    rms,
    rms_dbfs,
    rms_normalize,
)

SAMPLE_RATE = 16_000


def sine(amplitude: float = 1.0, freq: float = 200.0, duration_s: float = 1.0):
    t = torch.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * torch.sin(2 * math.pi * freq * t)).unsqueeze(0)


def test_rms_of_a_sine_is_amplitude_over_root_two():
    assert float(rms(sine(1.0))) == pytest.approx(1 / math.sqrt(2), abs=1e-3)


def test_full_scale_sine_reads_minus_three_dbfs():
    # dBFS is referenced to amplitude 1.0, not to the sine convention.
    assert rms_dbfs(sine(1.0)) == pytest.approx(-3.01, abs=0.01)
    assert peak_dbfs(sine(1.0)) == pytest.approx(0.0, abs=0.01)


def test_halving_the_amplitude_costs_six_db():
    assert rms_dbfs(sine(0.5)) - rms_dbfs(sine(1.0)) == pytest.approx(-6.02, abs=0.01)


def test_silence_reads_a_finite_floor_not_minus_infinity():
    silence = torch.zeros(1, 1000)
    assert rms_dbfs(silence) == SILENCE_DBFS
    assert peak_dbfs(silence) == SILENCE_DBFS


def test_rms_normalize_hits_the_target_from_either_direction():
    for amplitude in (0.01, 1.0):
        assert rms_dbfs(rms_normalize(sine(amplitude), -27.0)) == pytest.approx(
            -27.0, abs=1e-3
        )


def test_rms_normalize_removes_a_level_difference():
    quiet, loud = sine(0.1), sine(0.4)
    assert rms_dbfs(loud) - rms_dbfs(quiet) == pytest.approx(12.04, abs=0.01)
    normed_gap = rms_dbfs(rms_normalize(loud, -27.0)) - rms_dbfs(
        rms_normalize(quiet, -27.0)
    )
    assert normed_gap == pytest.approx(0.0, abs=1e-3)


def test_rms_normalize_preserves_waveform_shape():
    original = sine(0.2)
    normed = rms_normalize(original, -20.0)
    gain = float(normed[0, 100] / original[0, 100])
    assert torch.allclose(normed, original * gain, atol=1e-6)


def test_rms_normalize_leaves_silence_alone():
    silence = torch.zeros(1, 1000)
    assert torch.equal(rms_normalize(silence, -27.0), silence)


def test_rms_normalize_does_not_peak_limit():
    """Documented behaviour: an unreachable target clips rather than being capped.

    Clipping the signal under measurement would be a silent distortion, so the
    caller is expected to choose a target with headroom instead.
    """
    assert rms_normalize(sine(0.5), 0.0).abs().max() > 1.0


def test_peak_normalize_is_unchanged_by_the_additions():
    assert float(peak_normalize(sine(0.2), 0.9).abs().max()) == pytest.approx(0.9, abs=1e-4)


def test_rms_averages_over_channels():
    stereo = torch.cat([sine(1.0), torch.zeros(1, SAMPLE_RATE)], dim=0)
    # Mean square over both channels is half that of the sine alone.
    assert float(rms(stereo)) == pytest.approx(0.5, abs=1e-3)
