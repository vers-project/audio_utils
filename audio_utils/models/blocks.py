"""Shared Transformer building blocks.

Pre-norm Transformer with Rotary Position Embedding (RoPE) and PyTorch SDPA.
``causal=False`` (default) gives bidirectional attention.  ``key_padding_mask``
is forwarded to every layer so padded AudioBatch frames are never attended to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor

from audio_utils.data.audio_batch import AudioBatch


# ---------------------------------------------------------------------------
# Layer configuration
# ---------------------------------------------------------------------------


@dataclass
class LayerConfig:
    """Parameters for a single Transformer block.

    ``head_dim`` is the primary parameter; ``n_heads`` is derived so that
    it remains consistent when ``d_model`` varies across groups.

    Use :meth:`expand` to build a flat list from a config dict — it accepts
    both the legacy uniform format and the new per-group list format.
    """

    d_model:   int
    head_dim:  int
    mlp_ratio: int   = 4
    dropout:   float = 0.0
    attn_type: str   = "dot"   # "dot" (scaled dot-product) or "l2" (negative L2 distance)
    # Sliding-window attention: a query at frame t attends to keys in [t-w, t+w].
    # None = unrestricted, which is the previous behaviour and is reproduced exactly.
    #
    # Two independent properties, worth keeping apart because they want different sizes:
    #
    #   * a BOUNDED RECEPTIVE FIELD.  Stacking L windowed layers gives 1 + 2·Σw, so a
    #     per-frame output depends on a controllable span rather than on the whole
    #     sequence.  That is what turns a per-frame critic into a patch critic: a bad
    #     window can no longer be excused by a good one far away.  Wants w SMALL.
    #   * LENGTH GENERALISATION.  RoPE encodes relative position and its failure mode is
    #     relative distances never seen in training.  With a window, the largest distance
    #     any attention op computes is w, so a model trained on 2 s chunks meets no new
    #     distance on a 10 s sequence, and the softmax normalises over a fixed-size
    #     neighbourhood instead of over sequence length.  Wants only w ≤ what training
    #     contains -- and is *safer* the smaller w is.
    #
    # The second property degrades as w approaches the training chunk length: at T frames
    # a fraction (T − 2w)/T of training frames see a COMPLETE window.  On 2 s chunks
    # (T = 100) that is 96% at w = 2 and 0% at w = 50 -- where nearly every training frame
    # is a clipped boundary case while most test frames are not.  That is the same
    # train/test mismatch a conv receptive field wider than its chunk produces, so a large
    # window does not buy the safety it appears to.
    attn_window: int | None = None

    @property
    def n_heads(self) -> int:
        if self.d_model % self.head_dim != 0:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by head_dim={self.head_dim}"
            )
        return self.d_model // self.head_dim

    @staticmethod
    def expand(cfg: dict) -> list[LayerConfig]:
        """Parse a model config dict into a flat list of :class:`LayerConfig`.

        Accepts two formats:

        **Uniform** (backward-compatible)::

            d_model: 128
            head_dim: 32   # OR n_heads: 4
            n_layers: 6
            mlp_ratio: 4
            dropout: 0.0

        **Per-group** (new)::

            layers:
              - {d_model: 128, head_dim: 32, mlp_ratio: 4, count: 4}
              - {d_model: 256, head_dim: 32, mlp_ratio: 4, count: 2}

        The optional ``count`` key (default 1) repeats a group without
        duplicating YAML.  When ``head_dim`` is absent, it is derived from
        ``d_model // n_heads``.
        """
        if "layers" in cfg:
            result: list[LayerConfig] = []
            for spec in cfg["layers"]:
                spec  = dict(spec)
                count = spec.pop("count", 1)
                lc    = LayerConfig(**spec)
                result.extend([lc] * count)
            return result

        d_model = cfg["d_model"]
        if "head_dim" in cfg:
            head_dim = cfg["head_dim"]
        elif "n_heads" in cfg:
            head_dim = d_model // cfg["n_heads"]
        else:
            raise ValueError("config must contain 'head_dim' or 'n_heads'")

        lc = LayerConfig(
            d_model=d_model,
            head_dim=head_dim,
            mlp_ratio=cfg.get("mlp_ratio", 4),
            dropout=cfg.get("dropout", 0.0),
            attn_type=cfg.get("attn_type", "dot"),
            attn_window=cfg.get("attn_window"),
        )
        return [lc] * cfg["n_layers"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def masked_mean_pool(x: Tensor, mask: Tensor) -> Tensor:
    """Mean-pool over time, ignoring padding positions.

    Parameters
    ----------
    x    : (B, T, C)  BTC-format sequence.
    mask : (B, T)     True = valid step (not padding).

    Returns
    -------
    (B, C)
    """
    mask_f = mask.unsqueeze(-1).to(x.dtype)   # (B, T, 1)
    return (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _band_mask(
    n_frames: int, window: int, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Additive ``(1, 1, T, T)`` mask: 0 within ``|i − j| ≤ window``, ``-inf`` outside.

    Cached, because it depends only on the sequence length and the window — every layer
    and every step at a given length reuses the same tensor.
    """
    idx = torch.arange(n_frames, device=device)
    outside = (idx[:, None] - idx[None, :]).abs() > window
    return torch.zeros(
        (n_frames, n_frames), dtype=dtype, device=device
    ).masked_fill(outside, float("-inf"))[None, None]


class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE).

    Supports two score functions selected by ``attn_type``:

    * ``"dot"``  — standard scaled dot-product (via PyTorch SDPA).
    * ``"l2"``   — negative squared L2 distance, as proposed in ViT-GAN.
      Avoids the spectral-norm / attention-map collapse issue that occurs when
      dot-product attention is used inside a spectrally-normalised discriminator.

    Parameters
    ----------
    dim       : model dimensionality.
    n_heads   : number of attention heads.
    head_dim  : dimensionality per head.
    dropout   : attention dropout probability.
    causal    : causal vs. bidirectional attention.
    attn_type : ``"dot"`` or ``"l2"``.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
        attn_type: str = "dot",
        attn_window: int | None = None,
    ):
        super().__init__()
        if attn_type not in ("dot", "l2"):
            raise ValueError(f"attn_type must be 'dot' or 'l2', got {attn_type!r}")
        if attn_window is not None and attn_window < 0:
            raise ValueError(
                f"attn_window is a half-width in frames and must be >= 0, got "
                f"{attn_window}. Use None for unrestricted attention."
            )
        self.attn_window = attn_window
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.causal = causal
        self.attn_type = attn_type
        self.scale = head_dim ** -0.5

        self.rotary_emb = RotaryEmbedding(dim=head_dim, theta=50_000)
        self.qkv_proj = nn.Linear(dim, 3 * n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def _l2_attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        """Attention with negative-L2-distance scores.

        score_ij = -‖q_i − k_j‖² · scale
                 = (2 q·kᵀ − ‖q‖² − ‖k‖²) · scale

        Norms are preserved under rotation, so RoPE still encodes relative
        positions via the cross-term q·kᵀ.
        """
        # q, k, v: (B, H, T, head_dim)
        q_sq = (q * q).sum(-1, keepdim=True)                          # (B, H, T, 1)
        k_sq = (k * k).sum(-1, keepdim=True)                          # (B, H, T, 1)
        scores = (
            2.0 * torch.matmul(q, k.transpose(-2, -1))
            - q_sq
            - k_sq.transpose(-2, -1)
        ) * self.scale                                                  # (B, H, T, T)

        if attn_mask is not None:
            scores = scores + attn_mask
        if self.causal:
            T = q.size(-2)
            causal_mask = torch.full(
                (T, T), float("-inf"), device=q.device, dtype=q.dtype
            ).triu(1)
            scores = scores + causal_mask

        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout)
        return torch.matmul(attn, v)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        x                : (B, T, dim)
        key_padding_mask : (B, T) True = padding (ignored).
        returns            (B, T, dim)
        """
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            for t in qkv
        )  # each: (B, H, T, head_dim)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Padding mask → additive bias (-inf on ignored positions).
        # Only applied for bidirectional attention: combining is_causal=True
        # with an explicit attn_mask is not supported in all PyTorch versions.
        attn_mask = None
        if key_padding_mask is not None and not self.causal:
            attn_mask = torch.zeros(
                (B, 1, 1, T),
                dtype=x.dtype, device=x.device,
            ).masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        # Sliding window.  Additive like the padding mask, so both paths below take it
        # unchanged -- SDPA already accepts attn_mask, and _l2_attn already adds it to
        # its scores.  (Passing any explicit mask drops SDPA off the flash kernel, but
        # the padding mask above already does that whenever batches are ragged.)
        if self.attn_window is not None:
            band = _band_mask(T, self.attn_window, x.dtype, x.device)   # (1, 1, T, T)
            if attn_mask is None:
                attn_mask = band
            else:
                attn_mask = attn_mask + band                            # (B, 1, T, T)
                # A PADDED query whose whole window is also padding would now have an
                # all -inf row, and softmax of that is NaN -- which reaches VALID frames
                # in the next layer, because the padded position's value vector is NaN
                # and 0 * NaN = NaN in the attention matmul.  Re-opening the diagonal
                # keeps such rows finite.  Outputs at padded positions are meaningless
                # either way; this only stops them poisoning the rest.
                attn_mask.diagonal(dim1=-2, dim2=-1).zero_()

        if self.attn_type == "dot":
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
                scale=self.scale,
            )
        else:
            out = self._l2_attn(q, k, v, attn_mask)

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer block and encoder
# ---------------------------------------------------------------------------


def _stack_receptive_field(blocks) -> int | None:
    """``1 + 2·Σ w`` over a stack of TransformerBlocks, or ``None`` if any is global."""
    windows = [b.attn_window for b in blocks]
    if any(w is None for w in windows):
        return None
    return 1 + 2 * sum(windows)


class TransformerBlock(nn.Module):
    """Pre-norm block: LN → Attn → residual, LN → MLP → residual."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
        attn_type: str = "dot",
        attn_window: int | None = None,
    ):
        super().__init__()
        self.attn_window = attn_window
        self.ln1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, n_heads, head_dim, dropout=dropout, causal=causal,
                                  attn_type=attn_type, attn_window=attn_window)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(p=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + self.drop(self.attn(self.ln1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of TransformerBlocks forwarding ``key_padding_mask`` to each layer.

    Parameters
    ----------
    n_layers  : number of blocks.
    dim       : model dimensionality.
    n_heads   : number of attention heads.
    head_dim  : per-head dimensionality.  Defaults to ``dim // n_heads``.
    mlp_ratio : feedforward expansion factor.
    dropout   : dropout probability.
    causal    : causal vs. bidirectional attention.
    attn_type : ``"dot"`` or ``"l2"`` — score function for all blocks.
    """

    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
        head_dim: int | None = None,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
        attn_type: str = "dot",
        attn_window: int | None = None,
    ):
        super().__init__()
        hd = head_dim if head_dim is not None else dim // n_heads
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hd, mlp_ratio, dropout, causal, attn_type,
                             attn_window)
            for _ in range(n_layers)
        ])

    @property
    def receptive_field(self) -> int | None:
        """Frames one output depends on, or ``None`` if any layer is unrestricted."""
        return _stack_receptive_field(self.layers)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


# ---------------------------------------------------------------------------
# Shared sequence backbone
# ---------------------------------------------------------------------------


class SequenceBackbone(nn.Module):
    """Project → heterogeneous Transformer blocks → per-frame contextual features.

    Each block is configured independently via a :class:`LayerConfig`.  When
    ``d_model`` changes between consecutive groups a ``Linear`` transition is
    inserted automatically; identical widths use ``nn.Identity`` (zero cost).

    Always returns ``(B, T, output_dim)`` where ``output_dim`` is the
    ``d_model`` of the last block.  Use :attr:`output_dim` to wire the
    downstream projection correctly.

    Parameters
    ----------
    input_dim    : input feature width (e.g. 3 × n_freq for STFT features).
    layer_configs: flat list of per-layer configurations, typically produced
                   by :meth:`LayerConfig.expand`.
    """

    def __init__(self, input_dim: int, layer_configs: list[LayerConfig]):
        super().__init__()
        assert layer_configs, "layer_configs must not be empty"

        self.input_proj = nn.Linear(input_dim, layer_configs[0].d_model)
        self.input_norm = nn.LayerNorm(layer_configs[0].d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=lc.d_model,
                n_heads=lc.n_heads,
                head_dim=lc.head_dim,
                mlp_ratio=lc.mlp_ratio,
                dropout=lc.dropout,
                attn_type=lc.attn_type,
                attn_window=lc.attn_window,
            )
            for lc in layer_configs
        ])

        # Linear projection between groups with different d_model; Identity otherwise.
        self.transitions = nn.ModuleList([
            nn.Linear(a.d_model, b.d_model) if a.d_model != b.d_model else nn.Identity()
            for a, b in zip(layer_configs[:-1], layer_configs[1:])
        ])

        self.output_norm = nn.LayerNorm(layer_configs[-1].d_model)
        self._output_dim = layer_configs[-1].d_model

    @property
    def output_dim(self) -> int:
        """Feature width returned by :meth:`forward`."""
        return self._output_dim

    @property
    def receptive_field(self) -> int | None:
        """Frames one output frame depends on, or ``None`` if attention is unrestricted.

        ``1 + 2·Σ w`` over the blocks: each windowed layer extends reach by ``w`` on each
        side.  ``None`` as soon as one layer is unwindowed, since that layer alone makes
        every output depend on the whole sequence.
        """
        return _stack_receptive_field(self.blocks)

    def forward(self, batch: AudioBatch) -> Tensor:
        """
        batch  : AudioBatch  (B, D, T).
        returns  (B, T, output_dim).
        """
        x = self.input_norm(self.input_proj(batch.BTC))
        for i, block in enumerate(self.blocks):
            x = block(x, key_padding_mask=batch.key_padding_mask)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        return self.output_norm(x)
