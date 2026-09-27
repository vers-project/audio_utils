"""Tests for the dilated-convolution stack in ``audio_utils.models.conv``.

Two distinct groups of assertions, for two distinct reasons.

**Implementation correctness: batch-padding invariance.**  A sequence's
valid-frame outputs must not depend on how much padding the batch around it
happens to carry.  This holds only because every block re-zeroes padding before
its convolution — a plausible-looking implementation that masks once at the
input passes every shape check, trains without complaint, and silently corrupts
the last ``receptive_field / 2`` frames of every short sequence in a mixed
batch.  Note this is *not* what distinguishes a conv stack from a Transformer:
attention gets the same property for free from ``key_padding_mask``, and a RoPE
Transformer passes this test too.  It is here because the conv stack is the one
that can get it wrong.

**The actual motivation: boundedness and translation equivariance.**  These are
what a Transformer does not have.  An output frame depends on a fixed window
around it and on nothing else, however long the sequence is — so a stack trained
on 1-s chunks computes validation frames from exactly the evidence it was
trained on, whereas attention's relative distances and softmax normalisation
both change with sequence length.  ``receptive_field`` is also the number
configs are tuned against (it converts to milliseconds via the codec frame
rate), so it is measured here by autograd rather than trusted.
"""
from __future__ import annotations

import pytest
import torch

from audio_utils.data.audio_batch import AudioBatch
from audio_utils.models.conv import ConvBackbone, ConvEncoder, ConvLayerConfig


D_MODEL = 32
INPUT_DIM = 48


def make_configs(dilations=(1, 2, 4), kernel_size=3, **kw) -> list[ConvLayerConfig]:
    return ConvLayerConfig.expand(
        {
            "channels": D_MODEL,
            "kernel_size": kernel_size,
            "dilations": list(dilations),
            "mlp_ratio": 2,
            **kw,
        }
    )


# ---------------------------------------------------------------------------
# Receptive field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dilations, kernel_size, expected",
    [
        ((1,), 3, 3),
        ((1, 1, 1), 3, 7),
        ((1, 2, 4), 3, 15),
        ((1, 2, 4, 1, 2, 4), 3, 29),      # the VIC default: 29 frames @ 50 Hz = 580 ms
        ((1, 2, 4, 8), 3, 31),
        ((1, 2, 4), 5, 29),
    ],
)
def test_receptive_field_formula(dilations, kernel_size, expected):
    enc = ConvEncoder(make_configs(dilations, kernel_size))
    assert enc.receptive_field == expected


def test_receptive_field_matches_gradient():
    """Measure the receptive field by autograd and compare with the property.

    Only the frames within the receptive field of output position ``centre`` may
    have a non-zero gradient.  Weights are randomised away from init first: a
    freshly-built stack has near-zero contributions at the edge taps, which
    would make a too-small measured field look correct.
    """
    torch.manual_seed(0)
    enc = ConvEncoder(make_configs((1, 2, 4)))
    for p in enc.parameters():
        torch.nn.init.normal_(p, std=0.5)

    T = 81
    centre = T // 2
    x = torch.randn(1, T, D_MODEL, requires_grad=True)
    enc(x)[0, centre].sum().backward()

    touched = (x.grad.abs().sum(-1)[0] > 0).nonzero().flatten()
    measured = int(touched.max() - touched.min()) + 1
    assert measured == enc.receptive_field


# ---------------------------------------------------------------------------
# Length invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("film_dim", [None, 8])
def test_encoder_output_independent_of_batch_padding(film_dim):
    """The core claim: same sequence, alone vs. padded next to a longer one."""
    torch.manual_seed(0)
    enc = ConvEncoder(make_configs(), film_dim=film_dim).eval()

    short_len, long_len = 20, 60
    short = torch.randn(1, short_len, D_MODEL)
    long = torch.randn(1, long_len, D_MODEL)
    cond = torch.randn(2, film_dim) if film_dim is not None else None

    padded = torch.zeros(2, long_len, D_MODEL)
    padded[0, :short_len] = short[0]
    padded[1] = long[0]
    mask = torch.zeros(2, long_len, dtype=torch.bool)
    mask[0, short_len:] = True

    with torch.no_grad():
        batched = enc(padded, key_padding_mask=mask, cond=cond)
        alone = enc(short, cond=cond[:1] if cond is not None else None)

    torch.testing.assert_close(batched[0, :short_len], alone[0], rtol=0, atol=1e-6)


def test_encoder_zeroes_padded_positions():
    torch.manual_seed(0)
    enc = ConvEncoder(make_configs()).eval()

    x = torch.randn(2, 40, D_MODEL)
    mask = torch.zeros(2, 40, dtype=torch.bool)
    mask[0, 25:] = True

    with torch.no_grad():
        out = enc(x, key_padding_mask=mask)
    assert torch.count_nonzero(out[0, 25:]) == 0


def test_backbone_output_independent_of_batch_padding():
    """Same property end-to-end through ConvBackbone's AudioBatch interface."""
    torch.manual_seed(0)
    backbone = ConvBackbone(INPUT_DIM, make_configs()).eval()

    short_len, long_len = 20, 60
    short = torch.randn(INPUT_DIM, short_len)
    long = torch.randn(INPUT_DIM, long_len)

    together = AudioBatch.from_list([short, long], sample_rate=16_000)
    solo = AudioBatch.from_list([short], sample_rate=16_000)

    with torch.no_grad():
        out_together = backbone(together)
        out_solo = backbone(solo)

    torch.testing.assert_close(
        out_together[0, :short_len], out_solo[0, :short_len], rtol=0, atol=1e-6
    )


def test_backbone_is_translation_equivariant():
    """Shifting the signal shifts the features, away from the sequence edges.

    Distinct from the padding test: that one fixes the content and varies the
    batch, this one moves the content within a fixed-length sequence.  Frames
    within half a receptive field of either end are excluded, since zero-padding
    at a genuine boundary is real input, not an artefact.
    """
    torch.manual_seed(0)
    backbone = ConvBackbone(INPUT_DIM, make_configs()).eval()

    T, shift = 100, 17
    signal = torch.randn(INPUT_DIM, T - shift)
    early = torch.zeros(INPUT_DIM, T)
    late = torch.zeros(INPUT_DIM, T)
    early[:, : T - shift] = signal
    late[:, shift:] = signal

    batch = AudioBatch.from_list([early, late], sample_rate=16_000)
    with torch.no_grad():
        out = backbone(batch)

    margin = backbone.receptive_field // 2
    torch.testing.assert_close(
        out[0, margin : T - shift - margin],
        out[1, shift + margin : T - margin],
        rtol=0,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Configuration contract
# ---------------------------------------------------------------------------


def test_expand_reads_depth_from_dilations():
    configs = make_configs((1, 2, 4, 1, 2, 4))
    assert len(configs) == 6
    assert [c.dilation for c in configs] == [1, 2, 4, 1, 2, 4]
    assert all(c.d_model == D_MODEL and c.mlp_ratio == 2 for c in configs)


def test_groups_default_to_depthwise():
    depthwise = make_configs((1,))[0]
    assert depthwise.n_groups == D_MODEL
    assert make_configs((1,), groups=1)[0].n_groups == 1


@pytest.mark.parametrize(
    "cfg, message",
    [
        ({"dilations": [1]}, "channels"),
        ({"channels": D_MODEL}, "dilations"),
    ],
)
def test_expand_rejects_incomplete_config(cfg, message):
    with pytest.raises(ValueError, match=message):
        ConvLayerConfig.expand(cfg)


def test_even_kernel_rejected():
    with pytest.raises(ValueError, match="odd"):
        ConvLayerConfig(d_model=D_MODEL, kernel_size=4)


def test_non_divisible_groups_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ConvLayerConfig(d_model=D_MODEL, groups=7)


def test_film_and_cond_must_agree():
    enc_no_film = ConvEncoder(make_configs())
    with pytest.raises(ValueError, match="cond must be supplied"):
        enc_no_film(torch.randn(1, 10, D_MODEL), cond=torch.randn(1, 8))

    enc_film = ConvEncoder(make_configs(), film_dim=8)
    with pytest.raises(ValueError, match="cond must be supplied"):
        enc_film(torch.randn(1, 10, D_MODEL))


def test_film_is_identity_at_init():
    """A zero-initialised FiLM must leave a working configuration untouched."""
    torch.manual_seed(0)
    plain = ConvEncoder(make_configs()).eval()
    with_film = ConvEncoder(make_configs(), film_dim=8).eval()
    with_film.load_state_dict(plain.state_dict(), strict=False)

    x = torch.randn(2, 30, D_MODEL)
    with torch.no_grad():
        torch.testing.assert_close(
            plain(x), with_film(x, cond=torch.randn(2, 8)), rtol=0, atol=1e-6
        )


def test_encoder_rejects_mixed_widths():
    configs = make_configs((1, 2))
    configs[1] = ConvLayerConfig(d_model=D_MODEL * 2)
    with pytest.raises(ValueError, match="uniform d_model"):
        ConvEncoder(configs)
