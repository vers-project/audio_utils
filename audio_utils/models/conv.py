"""Shared dilated-convolution building blocks — the length-robust counterpart
to the Transformer stack in ``audio_utils.models.blocks``.

Why a convolutional stack
-------------------------
A RoPE Transformer is *relatively* positioned but not length-robust: training on
T-frame sequences never presents relative distances beyond T-1, and softmax
normalises over sequence length, so aggregating 200 keys differs from
aggregating 50 even at identical logits.  A convolutional stack has neither
property — its receptive field is fixed at construction and every output frame
is computed by the same translation-equivariant operator regardless of how long
the sequence is.

What is deliberately *not* changed
----------------------------------
:class:`ConvBlock` is :class:`~audio_utils.models.blocks.TransformerBlock` with
one substitution — the token-mixing operator::

    TransformerBlock:  x = x + attn(ln1(x));    x = x + mlp(ln2(x))
    ConvBlock:         x = x + dwconv(ln1(x));  x = x + mlp(ln2(x))

Same pre-norm placement, same double residual, same ``mlp_ratio`` FFN, same
dropout.  Attention becomes a dilated depthwise convolution and nothing else
moves, so a conv-vs-Transformer comparison at matched width isolates the mixing
operator rather than confounding it with a dozen incidental differences.

The convolution is depthwise by default (``groups=None`` → ``groups=d_model``),
which costs ``kernel_size × d_model`` parameters and leaves the parameter mass
in the channel MLP — exactly where the Transformer block carries it too.  Set
``groups=1`` for a full convolution when that ablation is wanted; no other code
changes.

Padding and masking
-------------------
Convolutions mix neighbouring frames, so padded positions would bleed into valid
ones within half a receptive field of every sequence end.  Every block therefore
zeroes padding *before* each convolution — after ``ln1``, since LayerNorm maps
zero-padding to non-zero values.  The residual branch still carries junk at
padded positions, but those positions are re-zeroed before the next convolution,
so no invalid value ever reaches a valid frame.

The property this buys is worth stating precisely, because it is the reason the
masking is done per-layer rather than once at the input:

    a sequence's valid-frame outputs are identical whether it is processed
    alone or padded alongside a longer one in the same batch.

``tests/test_conv.py`` asserts exactly that, and asserts that the measured
receptive field matches :attr:`ConvEncoder.receptive_field`.

Two stack-level classes, mirroring the two that already exist for Transformers:

======================  ==================  ====================================
Transformer             convolutional       contents
======================  ==================  ====================================
``TransformerEncoder``  ``ConvEncoder``     blocks only; caller owns the input
                                            projection (needed when something
                                            must be injected between projection
                                            and stack, e.g. conditioning)
``SequenceBackbone``    ``ConvBackbone``    ``input_proj`` + LN → blocks →
                                            ``output_norm``; consumes an
                                            ``AudioBatch`` directly
======================  ==================  ====================================
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from audio_utils.data.audio_batch import AudioBatch


__all__ = [
    "ConvLayerConfig",
    "FiLM",
    "ConvBlock",
    "ConvEncoder",
    "ConvBackbone",
]


# ---------------------------------------------------------------------------
# Layer configuration
# ---------------------------------------------------------------------------


@dataclass
class ConvLayerConfig:
    """Parameters for a single :class:`ConvBlock`.

    Mirrors :class:`~audio_utils.models.blocks.LayerConfig`, including the
    ``expand`` staticmethod, so both backbone families are configured the same
    way from YAML.

    Parameters
    ----------
    d_model     : channel width.
    kernel_size : temporal kernel width.  Odd values keep the output centred.
    dilation    : spacing between kernel taps.  The receptive field grows as
                  ``(kernel_size - 1) * dilation + 1`` per layer.
    groups      : convolution groups.  ``None`` (default) means depthwise
                  (``groups = d_model``); ``1`` means a full convolution.
    mlp_ratio   : FFN expansion ratio.
    dropout     : dropout probability.
    """

    d_model:     int
    kernel_size: int = 3
    dilation:    int = 1
    groups:      int | None = None
    mlp_ratio:   int = 4
    dropout:     float = 0.0

    def __post_init__(self):
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd so the convolution stays centred, "
                f"got {self.kernel_size}"
            )
        if self.d_model % self.n_groups != 0:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by groups={self.groups}"
            )

    @property
    def n_groups(self) -> int:
        """Resolved group count — ``d_model`` when ``groups`` is ``None``."""
        return self.d_model if self.groups is None else self.groups

    @property
    def receptive_field(self) -> int:
        """Frames this layer alone contributes to the receptive field."""
        return (self.kernel_size - 1) * self.dilation + 1

    @staticmethod
    def expand(cfg: dict) -> list[ConvLayerConfig]:
        """Parse a model config dict into a flat list of :class:`ConvLayerConfig`.

        The ``dilations`` list sets both the depth and the receptive field, so
        the two cannot silently disagree::

            channels: 224
            kernel_size: 3
            dilations: [1, 2, 4, 1, 2, 4]   # 6 layers, RF = 29 frames
            mlp_ratio: 4
            dropout: 0.0
            groups: null                    # null/absent → depthwise

        Every layer shares ``channels``; a stack that needs varying width should
        be built by constructing the list directly.
        """
        if "channels" not in cfg:
            raise ValueError("conv config must contain 'channels'")
        if "dilations" not in cfg:
            raise ValueError(
                "conv config must contain 'dilations' — the list sets both the "
                "depth and the receptive field"
            )
        return [
            ConvLayerConfig(
                d_model=cfg["channels"],
                kernel_size=cfg.get("kernel_size", 3),
                dilation=d,
                groups=cfg.get("groups"),
                mlp_ratio=cfg.get("mlp_ratio", 4),
                dropout=cfg.get("dropout", 0.0),
            )
            for d in cfg["dilations"]
        ]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _apply_mask(x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
    """Zero padded time steps.

    x                : (B, T, C)
    key_padding_mask : (B, T) True = padding.  ``None`` is a no-op, which is the
                       common case at train time where every chunk has the same
                       length.
    """
    if key_padding_mask is None:
        return x
    return x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


class FiLM(nn.Module):
    """Feature-wise linear modulation: ``x → γ(c) ⊙ x + β(c)``.

    A per-block conditioning path, as an alternative to concatenating the
    condition once at the input.  It matters more for a convolutional stack than
    for a Transformer: with a bounded receptive field there is no global route
    by which an input-time condition can reach every frame, so re-injecting it
    at each depth keeps it from being washed out.

    ``γ`` is initialised to 1 and ``β`` to 0 (zero-initialised output layer), so
    an untrained FiLM is the identity and adding it cannot destabilise a
    configuration that trained without it.

    Parameters
    ----------
    cond_dim : width of the conditioning vector.
    d_model  : width of the modulated features.
    """

    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        x    : (B, T, C)
        cond : (B, cond_dim) one condition for the whole sequence, or
               (B, T, cond_dim) a per-frame condition.
        returns (B, T, C)

        The per-frame form is what a bounded receptive field needs when the condition is
        itself time-varying: a global vector broadcast to every frame tells a window
        nothing about how its own content relates to the target, which is precisely the
        information a model that cannot see the whole sequence is missing.
        """
        h = self.proj(cond)
        if h.dim() == 2:
            h = h.unsqueeze(1)                                       # (B, 1, 2C), broadcast
        gamma, beta = h.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Pre-norm block: LN → (FiLM) → dilated conv → residual, LN → MLP → residual.

    Structurally identical to
    :class:`~audio_utils.models.blocks.TransformerBlock` with attention replaced
    by a dilated (by default depthwise) convolution — see the module docstring.

    Parameters
    ----------
    config   : the :class:`ConvLayerConfig` for this block.
    film_dim : when given, a :class:`FiLM` layer modulates the normalised input
               of the convolution branch, and ``forward`` requires ``cond``.
               ``None`` means no FiLM parameters exist at all.
    """

    def __init__(self, config: ConvLayerConfig, film_dim: int | None = None):
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.film = FiLM(film_dim, d_model) if film_dim is not None else None
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=config.kernel_size,
            dilation=config.dilation,
            groups=config.n_groups,
            padding=config.dilation * (config.kernel_size - 1) // 2,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p=config.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, config.mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.mlp_ratio * d_model, d_model),
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> Tensor:
        """
        x                : (B, T, C)
        key_padding_mask : (B, T) True = padding.
        cond             : (B, film_dim), required iff the block has FiLM.
        returns            (B, T, C)
        """
        if (self.film is None) != (cond is None):
            raise ValueError(
                "cond must be supplied exactly when the block was built with "
                f"film_dim (film={self.film is not None}, cond={cond is not None})"
            )

        h = self.ln1(x)
        if self.film is not None:
            h = self.film(h, cond)
        # Zero padding *after* the norm: LayerNorm maps zeros to non-zero values,
        # which the convolution would otherwise smear into neighbouring valid
        # frames.  This is what makes the block's output independent of how much
        # padding the batch happens to carry.
        h = _apply_mask(h, key_padding_mask)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(h)

        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Encoder and backbone
# ---------------------------------------------------------------------------


class ConvEncoder(nn.Module):
    """Stack of :class:`ConvBlock` s — the analogue of ``TransformerEncoder``.

    Takes and returns ``(B, T, C)``, leaving the input projection to the caller.
    Use this when something has to sit between the projection and the stack (a
    conditioning projection, for example); use :class:`ConvBackbone` otherwise.

    Parameters
    ----------
    layer_configs : per-layer configurations, typically from
                    :meth:`ConvLayerConfig.expand`.  All must share ``d_model``.
    film_dim      : condition width for per-block :class:`FiLM`, or ``None`` for
                    no FiLM parameters anywhere in the stack.
    """

    def __init__(
        self,
        layer_configs: list[ConvLayerConfig],
        film_dim: int | None = None,
    ):
        super().__init__()
        if not layer_configs:
            raise ValueError("layer_configs must not be empty")
        widths = {lc.d_model for lc in layer_configs}
        if len(widths) != 1:
            raise ValueError(
                f"ConvEncoder requires a uniform d_model across layers, got {sorted(widths)}"
            )

        self.layer_configs = layer_configs
        self.film_dim = film_dim
        self.blocks = nn.ModuleList(
            [ConvBlock(lc, film_dim=film_dim) for lc in layer_configs]
        )

    @property
    def d_model(self) -> int:
        return self.layer_configs[0].d_model

    @property
    def receptive_field(self) -> int:
        """Total receptive field in frames.

        ``1 + Σ (kernel_size - 1) · dilation`` over the stack: each layer adds
        its own taps on top of whatever the previous layers already reach.
        """
        return 1 + sum(lc.receptive_field - 1 for lc in self.layer_configs)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> Tensor:
        """
        x                : (B, T, C)
        key_padding_mask : (B, T) True = padding.
        cond             : (B, film_dim), required iff built with ``film_dim``.
        returns            (B, T, C), padded positions zeroed.
        """
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask, cond=cond)
        # Padded positions carry residual junk; zero them so callers (pooling,
        # losses, a downstream projection) never have to remember to.
        return _apply_mask(x, key_padding_mask)


class ConvBackbone(nn.Module):
    """Project → conv blocks → per-frame features: the analogue of
    :class:`~audio_utils.models.blocks.SequenceBackbone`.

    Consumes an :class:`~audio_utils.data.audio_batch.AudioBatch` and returns
    ``(B, T, output_dim)``, so it is a drop-in replacement wherever a
    ``SequenceBackbone`` is used as a feature extractor.

    Parameters
    ----------
    input_dim     : input feature width (e.g. the codec latent dimensionality).
    layer_configs : per-layer configurations, from :meth:`ConvLayerConfig.expand`.
    film_dim      : condition width for per-block FiLM, or ``None``.
    """

    def __init__(
        self,
        input_dim: int,
        layer_configs: list[ConvLayerConfig],
        film_dim: int | None = None,
    ):
        super().__init__()
        self.encoder = ConvEncoder(layer_configs, film_dim=film_dim)

        self.input_proj = nn.Linear(input_dim, self.encoder.d_model)
        self.input_norm = nn.LayerNorm(self.encoder.d_model)
        self.output_norm = nn.LayerNorm(self.encoder.d_model)

    @property
    def output_dim(self) -> int:
        """Feature width returned by :meth:`forward`."""
        return self.encoder.d_model

    @property
    def receptive_field(self) -> int:
        """Total receptive field in frames."""
        return self.encoder.receptive_field

    def forward(self, batch: AudioBatch, cond: Tensor | None = None) -> Tensor:
        """
        batch : AudioBatch (B, D, T).
        cond  : (B, film_dim), required iff built with ``film_dim``.
        returns (B, T, output_dim).
        """
        mask = batch.key_padding_mask
        x = self.input_norm(self.input_proj(batch.BTC))
        x = self.encoder(x, key_padding_mask=mask, cond=cond)
        return _apply_mask(self.output_norm(x), mask)
