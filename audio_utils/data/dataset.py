"""AudioDataset: loads waveform chunks from a CSV metadata file."""
from __future__ import annotations

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from audio_utils.data.transforms import (
    fixed_chunk,
    load_audio,
    load_audio_chunk,
    pad_to_multiple,
    peak_normalize,
    resample,
    select_channel,
)


class AudioDataset(Dataset):
    """Dataset that yields waveform chunks from a CSV metadata file.

    Only the requested chunk is decoded, so cost does not grow with file
    duration.  Chunks shorter than ``chunk_s`` (short files) are returned as
    they are — the collate function pads them and records the true length.

    Parameters
    ----------
    metadata           : DataFrame with at least a ``signal_path`` column
                         (already resolved to absolute paths via
                         ``resolve_paths``) and an optional ``channel`` column
                         (0-based; -1, NaN or absent → mono-mix all channels).
                         Optional ``start_s``/``end_s`` columns restrict each
                         row to a region of its file, so one file can supply
                         many rows — a VAD segment index, for instance.  NaN or
                         absent → that row spans its whole file, so a table
                         mixing segment-indexed and one-row-per-file corpora
                         behaves correctly for both.
    target_sr          : target sample rate; files are resampled on load.
    chunk_s            : chunk duration in seconds.  None = the full region.
    train              : if True, crop chunks randomly; otherwise from the start
                         of the region.
    pad_to_multiple_of : right-pad each chunk so its length is a multiple of
                         this many samples (e.g. a codec hop size).  None =
                         no padding.
    normalize_sequence : if True, peak-normalize the waveform before returning.
                         Leave False for models expecting raw amplitudes.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        target_sr: int,
        chunk_s: float | None = None,
        train: bool = True,
        pad_to_multiple_of: int | None = None,
        normalize_sequence: bool = False,
    ):
        self.paths = metadata["signal_path"].tolist()
        if "channel" in metadata.columns:
            self.channels = [
                int(c) if not pd.isna(c) else -1
                for c in metadata["channel"]
            ]
        else:
            self.channels = [-1] * len(self.paths)
        if "start_s" in metadata.columns and "end_s" in metadata.columns:
            # Per row, not per column: concatenating a segment-indexed corpus
            # with a one-row-per-file one leaves these columns present but NaN
            # on the file rows.  Without the guard ``float(nan)`` succeeds, so
            # ``end_s`` is never None, the whole-file path becomes unreachable
            # for those rows, and the failure surfaces much later as
            # ``round(nan * sr)`` inside a DataLoader worker.
            self.starts = [0.0 if pd.isna(s) else float(s) for s in metadata["start_s"]]
            self.ends = [None if pd.isna(e) else float(e) for e in metadata["end_s"]]
        else:
            self.starts = [0.0] * len(self.paths)
            self.ends = [None] * len(self.paths)
        self.target_sr = target_sr
        self.chunk_s = chunk_s
        self.train = train
        self.pad_to_multiple_of = pad_to_multiple_of
        self.normalize_sequence = normalize_sequence

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, index: int) -> Tensor:
        """Read, resample, channel-select and pad one item — without normalizing.

        Subclasses that need the raw amplitudes (to measure a level, say) call
        this, do their measurement, and normalize afterwards.
        """
        start_s, end_s = self.starts[index], self.ends[index]
        if self.chunk_s is None:
            duration_s = None if end_s is None else end_s - start_s
            wav, sr = load_audio(self.paths[index], start_s, duration_s)
        else:
            wav, sr = load_audio_chunk(
                self.paths[index],
                self.chunk_s,
                start_s=start_s,
                end_s=end_s,
                random_offset=self.train,
            )
        wav = resample(wav, sr, self.target_sr)
        wav = select_channel(wav, self.channels[index])
        if self.chunk_s is not None:
            wav = fixed_chunk(wav, self.target_sr, self.chunk_s)
        if self.pad_to_multiple_of is not None:
            wav = pad_to_multiple(wav, self.pad_to_multiple_of)
        return wav

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        wav = self._load(index)
        if self.normalize_sequence:
            wav = peak_normalize(wav)
        return {"wav": wav}  # (1, T)
