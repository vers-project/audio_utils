"""Tests for sliding-window attention in ``audio_utils.models.blocks``.

The window exists to give an attention stack the one property it otherwise lacks: a
**bounded receptive field**, so a per-frame output depends on a fixed span around it rather
than on the whole sequence.  A per-frame discriminator score is only a patch judgement if
that holds — otherwise every "frame score" is one global opinion read out at T positions,
and making those scores agree with each other fixes nothing.

Its second use is length generalisation.  RoPE's failure mode is relative distances never
seen in training; with a window, the largest distance any attention op computes is ``w``, so
a model trained on 2-s chunks meets none that are new on a longer sequence.

Three things are checked, in decreasing order of how quietly they would break:

* **No NaN from a fully-masked query row.**  New with the window: a padded query whose whole
  window is also padding has every key masked, and ``softmax`` of an all -inf row is NaN.
  That NaN reaches *valid* frames one layer later, because the padded position's value
  vector is NaN and ``0 * NaN = NaN`` in the attention matmul.  Nothing about the shapes or
  the loss would look wrong.
* **Batch-padding invariance**, the repo's standing rule: a sequence's valid outputs must not
  depend on how much padding the batch around it carries.
* **The receptive field is what the arithmetic says**, and unrestricted attention is
  untouched.
"""
import pytest
import torch

from audio_utils.data.audio_batch import AudioBatch
from audio_utils.models.blocks import (
    LayerConfig,
    RoPEAttention,
    SequenceBackbone,
    TransformerEncoder,
)

DIM, HEADS, HEAD_DIM = 32, 4, 8


def _encoder(window, n_layers=3, attn_type="dot", seed=0):
    torch.manual_seed(seed)
    return TransformerEncoder(
        n_layers, DIM, HEADS, HEAD_DIM, attn_type=attn_type, attn_window=window
    ).eval()


# --------------------------------------------------------------- receptive field


@pytest.mark.parametrize("window,n_layers,expected", [(2, 6, 25), (3, 6, 37), (1, 3, 7),
                                                      (8, 6, 97), (0, 4, 1)])
def test_receptive_field_is_one_plus_twice_the_summed_windows(window, n_layers, expected):
    assert _encoder(window, n_layers).receptive_field == expected


def test_unrestricted_attention_reports_no_receptive_field():
    assert _encoder(None).receptive_field is None


def test_one_global_layer_makes_the_whole_stack_global():
    """A single unwindowed layer makes every output depend on the whole sequence."""
    cfgs = [LayerConfig(DIM, HEAD_DIM, attn_window=2),
            LayerConfig(DIM, HEAD_DIM, attn_window=None),
            LayerConfig(DIM, HEAD_DIM, attn_window=2)]
    assert SequenceBackbone(DIM, cfgs).receptive_field is None
    cfgs[1] = LayerConfig(DIM, HEAD_DIM, attn_window=5)
    assert SequenceBackbone(DIM, cfgs).receptive_field == 1 + 2 * (2 + 5 + 2)


@pytest.mark.parametrize("attn_type", ["dot", "l2"])
def test_influence_reaches_exactly_as_far_as_the_receptive_field(attn_type):
    """Perturb frame 0; outputs must change within w·L frames of it and nowhere beyond.

    This is the property the whole feature exists for, and the only one a shape test
    cannot catch.  Both score functions are checked: they are separate code paths and the
    mask is applied in each of them independently.
    """
    window, n_layers = 2, 3
    enc = _encoder(window, n_layers, attn_type=attn_type)
    x = torch.randn(1, 40, DIM, dtype=torch.float64)
    enc = enc.double()

    with torch.no_grad():
        base = enc(x)
        perturbed = x.clone()
        # A *direction*, not a constant offset: every block starts with LayerNorm, which
        # subtracts the per-frame mean, so adding the same value to every channel is
        # erased before attention ever sees it and the frame would look uninfluential.
        perturbed[0, 0] = torch.randn(DIM, dtype=torch.float64) * 5.0
        moved = (enc(perturbed) - base).abs().amax(-1)[0]        # (T,)

    reach = window * n_layers                                     # frames each side
    assert (moved[: reach + 1] > 1e-9).all(), "influence stopped short of the field"
    assert (moved[reach + 1:] < 1e-9).all(), "influence escaped the receptive field"


# --------------------------------------------------------------- masking correctness


def test_a_window_at_least_the_sequence_length_is_plain_attention():
    """The window must be inert once it can no longer exclude anything."""
    x = torch.randn(2, 12, DIM)
    torch.manual_seed(0)
    unrestricted = _encoder(None, seed=0)
    torch.manual_seed(0)
    windowed = _encoder(11, seed=0)                               # T-1 reaches every key
    with torch.no_grad():
        assert torch.allclose(unrestricted(x), windowed(x), atol=1e-6)


@pytest.mark.parametrize("attn_type", ["dot", "l2"])
def test_a_fully_padded_query_window_does_not_produce_nan(attn_type):
    """The failure the diagonal guard exists for; without it this is silently NaN."""
    torch.manual_seed(0)
    attn = RoPEAttention(DIM, HEADS, HEAD_DIM, attn_type=attn_type, attn_window=1).eval()
    x = torch.randn(2, 24, DIM)
    key_padding_mask = torch.zeros(2, 24, dtype=torch.bool)
    key_padding_mask[1, 3:] = True                                # 21 padded frames
    with torch.no_grad():
        out = attn(x, key_padding_mask=key_padding_mask)
    assert torch.isfinite(out).all(), "all -inf row → NaN, and it spreads next layer"


@pytest.mark.parametrize("attn_type", ["dot", "l2"])
def test_valid_frames_do_not_depend_on_surrounding_padding(attn_type):
    """The repo's standing rule, re-checked because the window is a second mask."""
    torch.manual_seed(0)
    enc = _encoder(2, attn_type=attn_type)
    torch.manual_seed(1)
    short = torch.randn(1, 9, DIM)

    padded = torch.zeros(2, 25, DIM)
    padded[0, :9] = short[0]
    padded[1] = torch.randn(25, DIM)
    mask = torch.zeros(2, 25, dtype=torch.bool)
    mask[0, 9:] = True

    with torch.no_grad():
        alone = enc(short)
        batched = enc(padded, key_padding_mask=mask)
    assert torch.allclose(alone[0], batched[0, :9], atol=1e-5)


def test_a_negative_window_is_refused():
    with pytest.raises(ValueError, match="attn_window"):
        RoPEAttention(DIM, HEADS, HEAD_DIM, attn_window=-1)


def test_the_config_carries_the_window_through_expand():
    cfgs = LayerConfig.expand(
        {"d_model": DIM, "head_dim": HEAD_DIM, "n_layers": 4, "attn_window": 3}
    )
    assert [c.attn_window for c in cfgs] == [3, 3, 3, 3]
    assert LayerConfig.expand(
        {"d_model": DIM, "head_dim": HEAD_DIM, "n_layers": 2}
    )[0].attn_window is None
