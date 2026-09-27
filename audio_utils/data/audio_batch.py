"""AudioBatch: a padded batch of audio or latent tensors with length tracking."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AudioBatch:
    """Padded batch of audio or latent tensors.

    Internal storage is always (B, C, T_max) — the natural shape produced
    by torchaudio and codec encoders.  Two explicit views expose the data
    in whichever shape downstream code needs:

        batch.BCT  →  (B, C, T)   for Conv1d, codec encode/decode
        batch.BTC  →  (B, T, C)   for nn.TransformerEncoder, attention, linear

    Padding convention
    ------------------
    lengths[i] is the number of valid time steps for sample i.
    Samples are right-padded with zeros beyond that length.

        batch.padding_mask      (B, T)  True  = valid step
        batch.key_padding_mask  (B, T)  True  = padding (ignored step)
            → pass directly to nn.MultiheadAttention / TransformerEncoder
    """

    data: Tensor      # (B, C, T_max)  canonical internal storage
    lengths: Tensor   # (B,)           valid T per sample, dtype=torch.long
    sample_rate: int

    # ------------------------------------------------------------------
    # Shape views
    # ------------------------------------------------------------------

    @property
    def BCT(self) -> Tensor:
        """(B, C, T) view — for codec ops and Conv1d layers."""
        return self.data

    @property
    def BTC(self) -> Tensor:
        """(B, T, C) view — for Transformer / attention / linear layers."""
        return self.data.transpose(1, 2)

    # ------------------------------------------------------------------
    # Padding masks
    # ------------------------------------------------------------------

    @property
    def padding_mask(self) -> Tensor:
        """(B, T) bool, True where the time step is valid (not padding)."""
        T = self.data.shape[-1]
        idx = torch.arange(T, device=self.data.device)
        return idx.unsqueeze(0) < self.lengths.unsqueeze(1)

    @property
    def key_padding_mask(self) -> Tensor:
        """(B, T) bool, True where the step should be IGNORED.

        Pass directly as ``src_key_padding_mask`` to
        ``nn.MultiheadAttention`` or ``nn.TransformerEncoder``.
        """
        return ~self.padding_mask

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def max_length(self) -> int:
        return self.data.shape[2]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_list(cls, sequences: list[Tensor], sample_rate: int) -> AudioBatch:
        """Build a padded AudioBatch from a list of (C, T_i) tensors.

        Shorter sequences are right-padded with zeros to match the longest
        one in the list.
        """
        if not sequences:
            raise ValueError("Cannot create AudioBatch from an empty list.")
        lengths = torch.tensor([s.shape[-1] for s in sequences], dtype=torch.long)
        T_max = int(lengths.max().item())
        B, C = len(sequences), sequences[0].shape[0]
        padded = sequences[0].new_zeros(B, C, T_max)
        for i, s in enumerate(sequences):
            padded[i, :, : s.shape[-1]] = s
        return cls(data=padded, lengths=lengths, sample_rate=sample_rate)

    # ------------------------------------------------------------------
    # Unbatching
    # ------------------------------------------------------------------

    def unbatch(self) -> list[Tensor]:
        """Return a list of (C, T_i) tensors with padding stripped."""
        return [
            self.data[i, :, : int(self.lengths[i].item())]
            for i in range(self.batch_size)
        ]

    # ------------------------------------------------------------------
    # Frame-space conversion
    # ------------------------------------------------------------------

    def as_frames(self, frame_data: Tensor, frame_lengths: Tensor) -> AudioBatch:
        """Return a new AudioBatch in codec frame space.

        Typically called by a codec wrapper after running the encoder:

            z = encoder(batch.BCT)                       # (B, D, T_frames)
            frame_lengths = batch.lengths // hop_size
            latent_batch = batch.as_frames(z, frame_lengths)

        Parameters
        ----------
        frame_data    : (B, D, T_frames_max) encoder output (may be padded).
        frame_lengths : (B,) valid frame count per sample.
        """
        return AudioBatch(
            data=frame_data,
            lengths=frame_lengths.to(dtype=torch.long, device=frame_data.device),
            sample_rate=self.sample_rate,
        )

    # ------------------------------------------------------------------
    # Device / memory
    # ------------------------------------------------------------------

    def detach(self) -> AudioBatch:
        return AudioBatch(
            data=self.data.detach(),
            lengths=self.lengths,
            sample_rate=self.sample_rate,
        )

    def to(self, *args, **kwargs) -> AudioBatch:
        return AudioBatch(
            data=self.data.to(*args, **kwargs),
            lengths=self.lengths.to(*args, **kwargs),
            sample_rate=self.sample_rate,
        )

    def pin_memory(self) -> AudioBatch:
        return AudioBatch(
            data=self.data.pin_memory(),
            lengths=self.lengths.pin_memory(),
            sample_rate=self.sample_rate,
        )

    def __repr__(self) -> str:
        return (
            f"AudioBatch(shape={tuple(self.data.shape)}, "
            f"lengths={self.lengths.tolist()}, sr={self.sample_rate})"
        )
