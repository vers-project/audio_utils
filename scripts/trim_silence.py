"""Trim leading and trailing silence from audio files described in a metadata CSV.

Reads a CSV with a ``signal_path`` column (relative to ``data.dataset_root``),
trims silence from each file, writes the trimmed files under
``dataset_root/output_subdir/`` preserving the original directory structure,
and saves an updated metadata CSV at ``output_dir/metadata.csv``.  Signal
paths in the output CSV are relative to ``dataset_root`` (e.g.
``output_subdir/speaker/file.wav``), so calibration paths and other relative
columns in the original CSV remain valid without change.

Silence detection matches the sox silence effect:
    sox in.wav out.wav silence 1 <duration> <threshold>%  reverse  \\
                        silence 1 <duration> <threshold>%  reverse

Implementation details (from sox 14.4.2 silence.c):
  - RMS window  : sample_rate // 50 samples (~20 ms), sliding sample-by-sample
  - Threshold   : percentage of full scale (float audio in [-1,1] → 1% = 0.01)
  - Duration    : sustained above-threshold samples = round(duration_s * sample_rate)
  - Trailing    : mirrored algorithm on the reversed signal (= reverse silence reverse)

Default parameters (threshold_pct=1.0, duration=0.1) reproduce the common
sox invocation used in many speech processing papers:
    sox in.wav out.wav silence 1 0.1 1% reverse silence 1 0.1 1% reverse

Usage
-----
    uv run --extra cpu scripts/trim_silence.py \\
        --config path/to/trim_silence.yaml \\
        --output-dir /data/ravdess_trimmed

Config keys
-----------
    data.input_csv          : path to the input metadata CSV
    data.dataset_root       : root directory of the source audio files
    data.output_subdir      : subdirectory name under dataset_root for trimmed audio
    silence.threshold_pct   : silence threshold as % of full scale (default: 1.0)
    silence.duration        : minimum above-threshold duration in seconds (default: 0.1)
    skip_existing           : skip output files that already exist (default: false)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchaudio
from experiment_launcher import parse_args
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Silence trimming — matches sox silence effect
# ---------------------------------------------------------------------------

def _first_sustained_run(above: torch.Tensor, run_len: int) -> int:
    """Return the first index i such that above[i : i+run_len] is all True.

    Returns len(above) when no such run exists (audio is entirely silent).
    """
    n = above.shape[0]
    if run_len <= 0:
        return 0
    if run_len > n:
        return n

    above_f = above.float()
    # padded[i] = sum(above_f[0:i])
    padded = torch.cat([torch.zeros(1, dtype=above_f.dtype), above_f.cumsum(0)])
    # window_sum[i] = sum(above_f[i : i+run_len])
    window_sum = padded[run_len:] - padded[:-run_len]
    valid = (window_sum >= run_len).nonzero(as_tuple=False)
    return int(valid[0].item()) if valid.numel() > 0 else n


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold_pct: float = 1.0,
    duration: float = 0.1,
) -> torch.Tensor:
    """Trim leading and trailing silence, matching:
        sox in.wav out.wav silence 1 <duration> <threshold_pct>%  reverse
                            silence 1 <duration> <threshold_pct>%  reverse

    Parameters
    ----------
    waveform:       (C, T) float tensor, audio in [-1, 1]
    sample_rate:    samples per second
    threshold_pct:  silence threshold as percentage of full scale (sox % unit)
    duration:       minimum above-threshold duration in seconds (sox duration arg)

    Returns the original waveform unchanged when it is entirely silent.
    """
    # sox: window_size = (sample_rate / 50) * channels  — for the RMS window.
    # We operate on mono (mean across channels) so window_size = sample_rate // 50.
    window = max(1, sample_rate // 50)

    # sox %: scaled_value = rms / SOX_SAMPLE_MAX * 100; compare > threshold_pct.
    # For float32 audio in [-1, 1]: SOX_SAMPLE_MAX normalises to 1.0.
    threshold = threshold_pct / 100.0

    # sox: start_duration in samples = round(duration_seconds * sample_rate)
    duration_samples = max(1, round(duration * sample_rate))

    mono = waveform.mean(dim=0).double()  # (T,) — double for RMS precision
    T = mono.shape[0]

    if T < window:
        return waveform

    # Sliding window RMS, sample-by-sample (sox compute_rms / update_rms).
    # rms[i] = sqrt( mean( mono[i : i+window]^2 ) ),  length = T - window + 1.
    sq = mono.pow(2)
    padded_cs = torch.cat([torch.zeros(1, dtype=sq.dtype), sq.cumsum(0)])
    window_sum = padded_cs[window:] - padded_cs[:-window]   # (T - window + 1,)
    rms = (window_sum / window).sqrt().float()

    above = rms > threshold   # bool, length n = T - window + 1
    n = above.shape[0]

    # Leading silence: first RMS index that begins a sustained run of
    # duration_samples above-threshold samples (sox SILENCE_TRIM state).
    start_rms = _first_sustained_run(above, duration_samples)

    # Trailing silence: mirror of the same algorithm on the reversed signal
    # (sox: reverse silence reverse).  j = first run start in flipped RMS;
    # end_rms = n - j is the exclusive end in forward RMS space.
    j = _first_sustained_run(above.flip(0), duration_samples)
    end_rms = n - j

    if start_rms >= end_rms:
        return waveform  # entirely silent

    # Map RMS indices back to sample indices.
    # rms[i] covers samples [i, i+window).  The last window in the kept region
    # starts at end_rms - 1, so the last kept sample is end_rms - 1 + window - 1.
    start_sample = start_rms
    end_sample   = min(end_rms + window, T)

    return waveform[:, start_sample:end_sample]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@parse_args
def main(config: dict, output_dir: Path) -> None:
    data_cfg      = config["data"]
    input_csv     = Path(data_cfg["input_csv"])
    dataset_root  = Path(data_cfg["dataset_root"])
    output_subdir = data_cfg["output_subdir"]

    sil_cfg       = config.get("silence", {})
    threshold_pct = float(sil_cfg.get("threshold_pct", 1.0))
    duration      = float(sil_cfg.get("duration", 0.1))
    skip_existing = bool(config.get("skip_existing", False))

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    df = pd.read_csv(input_csv)
    if "signal_path" not in df.columns:
        raise SystemExit("Input CSV must have a 'signal_path' column.")

    audio_out  = dataset_root / output_subdir
    output_csv = output_dir / "metadata.csv"
    audio_out.mkdir(parents=True, exist_ok=True)

    new_paths: list[str] = []
    skipped = done = errors = 0

    pbar = tqdm(df.iterrows(), total=len(df), unit="file")
    for _, row in pbar:
        rel_path = row["signal_path"]   # relative to dataset_root
        src_path = dataset_root / rel_path
        dst_rel  = str(Path(output_subdir) / rel_path)  # relative to dataset_root
        dst_path = dataset_root / dst_rel

        if skip_existing and dst_path.exists():
            new_paths.append(dst_rel)
            skipped += 1
            pbar.set_postfix(done=done, skipped=skipped, errors=errors)
            continue

        if not src_path.exists():
            tqdm.write(f"  WARNING: source not found, skipping: {src_path}")
            new_paths.append(dst_rel)
            errors += 1
            pbar.set_postfix(done=done, skipped=skipped, errors=errors)
            continue

        try:
            waveform, sr = torchaudio.load(str(src_path))
            trimmed = trim_silence(waveform, sr, threshold_pct, duration)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(dst_path), trimmed, sr)
            new_paths.append(dst_rel)
            done += 1
        except Exception as exc:
            tqdm.write(f"  ERROR processing {src_path}: {exc}")
            new_paths.append(dst_rel)
            errors += 1
        pbar.set_postfix(done=done, skipped=skipped, errors=errors)

    out_df = df.copy()
    out_df["signal_path"] = new_paths
    out_df.to_csv(output_csv, index=False)

    print(f"Done. {done} trimmed, {skipped} skipped, {errors} errors.")
    print(f"Output audio : {audio_out}/")
    print(f"Output CSV   : {output_csv}")


if __name__ == "__main__":
    main()
