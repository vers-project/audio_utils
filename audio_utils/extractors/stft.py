"""STFT-domain feature extractor and synthesizer.

STFTExtractor  : waveform → [magnitude, cos_phase, sin_phase] per frequency bin.
STFTSynthesizer: raw decoder predictions → waveform, plus the STFT-domain loss.

The [mag, cos, sin] triplet per bin is a natural symmetric representation:
the extractor produces it from the complex STFT, and the synthesizer accepts
raw predictions [raw_mag, raw_cos, raw_sin] that it converts via
    magnitude = exp(raw_mag)              (ensures positivity)
    phase     = [raw_cos, raw_sin] / ‖…‖  (forces unit circle)
before reconstructing the complex spectrum and applying iSTFT.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.functional as AF
from torch import Tensor

from audio_utils.data.audio_batch import AudioBatch


def _mel_filterbank_weights(n_freq: int, n_mels: int, sample_rate: int) -> Tensor:
    """Per-STFT-bin importance weights derived from a triangular mel filterbank.

    Each weight is the sum of mel filter responses at that bin, normalized so
    the mean equals 1.  Low-frequency bins receive higher weights, reflecting
    human perceptual sensitivity.

    Returns
    -------
    Tensor of shape (n_freq,), dtype float32.
    """
    fb = AF.melscale_fbanks(
        n_freqs=n_freq,
        f_min=0.0,
        f_max=float(sample_rate) / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )                                          # (n_freq, n_mels)
    weights = fb.sum(dim=1)                    # (n_freq,)
    return weights / weights.mean().clamp(min=1e-8)


class STFTExtractor(nn.Module):
    """Compute STFT and expose per-bin [magnitude, cos, sin] features.

    Parameters
    ----------
    n_fft         : FFT size.
    hop_length    : hop size in samples.
    win_length    : window length.  Defaults to n_fft.
    sample_rate   : audio sample rate (stored in output AudioBatch metadata).
    center        : whether to center-pad the signal (matches librosa convention).
    log_magnitude : if True, replace the linear magnitude channel with
                    ``log(mag.clamp(1e-8))`` before returning features.
                    Phase (cos/sin) is always computed from the linear magnitude
                    regardless of this flag.  Enable for discriminators to avoid
                    low-frequency dominance (same motivation as ``log_scale=True``
                    in the reconstruction loss).
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        win_length: int | None = None,
        center: bool = True,
        log_magnitude: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.sample_rate = sample_rate
        self.center = center
        self.log_magnitude = log_magnitude
        self.n_freq = n_fft // 2 + 1
        self.register_buffer("_window", torch.hann_window(self.win_length))

    @property
    def output_dim(self) -> int:
        """Feature dimensionality per frame: 3 × n_freq."""
        return 3 * self.n_freq

    def _stft(self, wav: Tensor) -> Tensor:
        """wav: (B, T) → complex (B, n_freq, T_frames)."""
        return torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window,
            center=self.center,
            return_complex=True,
        )

    def forward(self, wav_batch: AudioBatch) -> AudioBatch:
        """Extract STFT features from a batch of waveforms.

        Parameters
        ----------
        wav_batch : AudioBatch  (B, 1, T_samples).

        Returns
        -------
        AudioBatch  (B, 3 * n_freq, T_frames)  with frame-level lengths.
        """
        wav = wav_batch.BCT.squeeze(1)               # (B, T)
        X = self._stft(wav)                          # complex (B, n_freq, T_frames)
        T_frames = X.shape[-1]

        mag = X.abs()                                # (B, n_freq, T_frames)
        norm = mag.clamp(min=1e-8)                   # always linear for phase
        cos = X.real / norm
        sin = X.imag / norm

        if self.log_magnitude:
            mag = norm.log()                         # reuse clamped value

        features = torch.cat([mag, cos, sin], dim=1) # (B, 3*n_freq, T_frames)

        frame_lengths = (wav_batch.lengths // self.hop_length + 1).clamp(max=T_frames)
        return AudioBatch(
            data=features,
            lengths=frame_lengths,
            sample_rate=self.sample_rate,
        )


class MultiResolutionSTFTLoss(nn.Module):
    """Waveform-level L1 reconstruction loss at multiple STFT resolutions.

    For each resolution an ``STFTExtractor`` (with ``log_magnitude=True``)
    extracts ``[log_mag, cos, sin]`` features from both the reconstructed and
    target waveforms.  The L1 distance is averaged over the feature and time
    dimensions, then averaged across all resolutions.

    Using several resolutions simultaneously supervises the model at multiple
    time-frequency scales:

    - Small ``n_fft`` (e.g. 256)  → fine temporal resolution, captures transients.
    - Large ``n_fft`` (e.g. 2048) → fine spectral resolution, captures harmonics.

    This loss operates on waveforms and is agnostic to the decoder's native STFT
    resolution.  It is intended as a complement to the decoder-domain
    reconstruction loss, not a replacement.

    Parameters
    ----------
    resolutions : list of dicts, each with keys ``n_fft``, ``hop_length``, and
                  optionally ``win_length`` (defaults to ``n_fft``).
    sample_rate : audio sample rate — passed to each ``STFTExtractor``.
    """

    def __init__(self, resolutions: list[dict], sample_rate: int):
        super().__init__()
        self.extractors = nn.ModuleList([
            STFTExtractor(
                n_fft=r["n_fft"],
                hop_length=r["hop_length"],
                win_length=r.get("win_length", r["n_fft"]),
                sample_rate=sample_rate,
                log_magnitude=True,
            )
            for r in resolutions
        ])

    def forward(self, wav_recon: AudioBatch, wav_target: AudioBatch) -> Tensor:
        """Compute multi-resolution STFT loss.

        Parameters
        ----------
        wav_recon  : AudioBatch  (B, 1, T) — reconstructed waveform.
        wav_target : AudioBatch  (B, 1, T) — target waveform.

        Returns
        -------
        Scalar L1 loss averaged over resolutions, frequency bins, and valid frames.
        """
        total = wav_recon.BCT.new_zeros(())
        for extractor in self.extractors:
            recon_feat  = extractor(wav_recon)   # (B, 3*n_freq, T_frames)
            target_feat = extractor(wav_target)  # (B, 3*n_freq, T_frames)

            mask    = target_feat.padding_mask.float()           # (B, T_frames)
            n_valid = mask.sum().clamp(min=1)

            diff = (recon_feat.BCT - target_feat.BCT).abs()     # (B, 3*n_freq, T_frames)
            loss = (diff.mean(dim=1) * mask).sum() / n_valid    # scalar
            total = total + loss

        return total / len(self.extractors)


class STFTConsistencyLoss(nn.Module):
    """Self-consistency regulariser for STFT-domain codec predictions.

    A predicted complex spectrogram ``X_pred`` is *consistent* iff::

        STFT(ISTFT(X_pred)) = X_pred

    i.e. it lies in the image of the STFT operator and can be realised as a
    waveform without overlap-add artefacts.  Frame-by-frame predictions that
    violate this constraint produce audible discontinuities at every hop
    boundary — typically perceived as a rapid, regular buzzing noise at the
    hop rate.

    This loss compares:

    * ``roundtrip`` = ``STFTExtractor(wav_recon)`` — STFT of the waveform
    * ``pred_stft`` = ``STFTSynthesizer.raw_to_stft_features(pred_raw)``

    and penalises their L1 distance.  No target waveform is involved: the
    loss is purely self-referential, making it compatible with lossy codecs
    where exact waveform reconstruction is impossible.

    The STFT parameters must match those of the synthesizer that produced
    ``wav_recon``.

    Parameters
    ----------
    n_fft, hop_length, win_length, sample_rate : STFT parameters, must
        match the codec synthesizer's settings exactly.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        win_length: int | None = None,
    ):
        super().__init__()
        self.extractor = STFTExtractor(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            sample_rate=sample_rate,
            log_magnitude=True,
        )

    def forward(self, wav_recon: AudioBatch, pred_stft: AudioBatch) -> Tensor:
        """Compute the STFT consistency loss.

        Parameters
        ----------
        wav_recon : AudioBatch  (B, 1, T_samples) — iSTFT waveform from synthesizer.
        pred_stft : AudioBatch  (B, 3 * n_freq, T_frames) — decoder predictions
                    in [log_mag, cos, sin] format, as returned by
                    ``STFTSynthesizer.raw_to_stft_features``.

        Returns
        -------
        Scalar L1 consistency loss.
        """
        roundtrip = self.extractor(wav_recon)           # (B, 3*n_freq, T_frames)

        # Frame counts should be equal in practice; clip defensively.
        T   = min(roundtrip.BCT.shape[-1], pred_stft.BCT.shape[-1])
        rt  = roundtrip.BCT[..., :T]
        ps  = pred_stft.BCT[..., :T]

        mask    = pred_stft.padding_mask[:, :T].float()  # (B, T)
        n_valid = mask.sum().clamp(min=1)

        diff = (rt - ps).abs()                           # (B, 3*n_freq, T)
        return (diff.mean(dim=1) * mask).sum() / n_valid


class MagnitudeLaplacianLoss(nn.Module):
    """Temporal difference penalty on the predicted log-magnitude spectrogram.

    Penalises frame-to-frame oscillations in the magnitude without penalising
    gradual spectral changes or isolated transients:

    * **Order 1** (Total Variation): penalises ``|log|X̂(t+1)| − log|X̂(t)||``.
      Consonant onsets are penalised proportionally to their amplitude.

    * **Order 2** (Laplacian): penalises
      ``|log|X̂(t+2)| − 2·log|X̂(t+1)| + log|X̂(t)||``.
      A ramp (gradual onset) has zero second derivative → zero penalty.
      A step (plosive) has a non-zero second derivative at only two frames
      (onset and offset) → minimal penalty.
      An alternating pattern h, l, h, l, … has a large second derivative at
      *every* frame → strong penalty.  Order 2 is therefore much more
      consonant-friendly than order 1 for targeting the 50 Hz buzzing artifact.

    The ``norm`` argument controls whether differences are penalised with L1
    (absolute value) or L2 (squared).  L1 treats all frames equally regardless
    of the magnitude of the deviation, so it is more aggressive at eliminating
    many small oscillations while being less destructive toward the occasional
    large transient.

    Parameters
    ----------
    n_freq : number of STFT frequency bins (``n_fft // 2 + 1``).
    order  : ``1`` (first-order difference) or ``2`` (second-order / Laplacian).
    norm   : ``"l1"`` or ``"l2"``.
    """

    def __init__(self, n_freq: int, order: int = 2, norm: str = "l1"):
        super().__init__()
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")
        self.n_freq = n_freq
        self.order  = order
        self.norm   = norm

    def forward(self, pred_raw: AudioBatch) -> Tensor:
        """Compute the magnitude temporal difference loss.

        Parameters
        ----------
        pred_raw : AudioBatch  (B, 3 * n_freq, T_frames) — raw decoder output.
                   The first ``n_freq`` channels are raw log-magnitudes.

        Returns
        -------
        Scalar loss.
        """
        log_mag = pred_raw.BCT[:, :self.n_freq, :]          # (B, n_freq, T)

        if self.order == 1:
            diff = log_mag[..., 1:] - log_mag[..., :-1]     # (B, n_freq, T-1)
            # frame t is valid iff t and t+1 are valid; checking t+1 suffices
            mask = pred_raw.padding_mask[:, 1:].float()      # (B, T-1)
        else:
            diff = (log_mag[..., 2:]
                    - 2 * log_mag[..., 1:-1]
                    + log_mag[..., :-2])                     # (B, n_freq, T-2)
            # frame t is valid iff t, t+1, t+2 are valid; checking t+2 suffices
            mask = pred_raw.padding_mask[:, 2:].float()      # (B, T-2)

        diff    = diff.abs() if self.norm == "l1" else diff.pow(2)
        n_valid = mask.sum().clamp(min=1)
        return (diff.mean(dim=1) * mask).sum() / n_valid


class STFTSynthesizer(nn.Module):
    """Convert raw decoder predictions to waveforms and compute the training loss.

    Raw predictions are [raw_mag, raw_cos, raw_sin] per frequency bin.
    Before iSTFT, activations are applied:
        magnitude = exp(raw_mag)
        phase     = [raw_cos, raw_sin] / ‖[raw_cos, raw_sin]‖₂

    Parameters
    ----------
    n_fft, hop_length, win_length, sample_rate, center : same as STFTExtractor.
    n_mels : number of mel filters used to derive frequency-bin weights for
             ``mel_weighted_reconstruction_loss``.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        win_length: int | None = None,
        center: bool = True,
        n_mels: int = 80,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.sample_rate = sample_rate
        self.center = center
        self.n_freq = n_fft // 2 + 1
        self.register_buffer("_window", torch.hann_window(self.win_length))
        self.register_buffer(
            "_mel_weights",
            _mel_filterbank_weights(self.n_freq, n_mels, sample_rate),
        )

    @property
    def input_dim(self) -> int:
        """Expected raw feature dimensionality per frame: 3 × n_freq."""
        return 3 * self.n_freq

    def raw_to_stft_features(self, raw_batch: AudioBatch) -> AudioBatch:
        """Convert raw decoder predictions to log-magnitude STFT features.

        Applies activations (exp for magnitude, L2-normalisation for phase)
        but does *not* run iSTFT.  The resulting ``[log_mag, cos, sin]``
        format matches ``STFTExtractor(log_magnitude=True)``, making it
        directly comparable to the re-analysed spectrogram of the
        reconstructed waveform.

        Used as input to ``STFTConsistencyLoss``.
        """
        F   = self.n_freq
        raw = raw_batch.BCT                              # (B, 3*n_freq, T)
        raw_mag = raw[:, :F]
        raw_cos = raw[:, F:2 * F]
        raw_sin = raw[:, 2 * F:]

        phase_norm = (raw_cos.pow(2) + raw_sin.pow(2)).clamp(min=1e-8).sqrt()
        features = torch.cat(
            [raw_mag, raw_cos / phase_norm, raw_sin / phase_norm], dim=1
        )
        return AudioBatch(data=features, lengths=raw_batch.lengths, sample_rate=self.sample_rate)

    # ------------------------------------------------------------------
    # Complex spectrum helpers
    # ------------------------------------------------------------------

    def _features_to_complex(self, features: Tensor) -> Tensor:
        """Convert extractor output features to a complex STFT.

        Parameters
        ----------
        features : (B, 3 * n_freq, T)  [mag, cos, sin] — cos/sin already
                   on the unit circle (direct output of STFTExtractor).

        Returns
        -------
        complex Tensor (B, n_freq, T).
        """
        F = self.n_freq
        mag = features[:, :F, :]
        cos = features[:, F:2 * F, :]
        sin = features[:, 2 * F:, :]
        return torch.complex(mag * cos, mag * sin)

    def _raw_to_complex(self, raw: Tensor) -> Tensor:
        """Apply activations to raw decoder output and build a complex STFT.

        Parameters
        ----------
        raw : (B, 3 * n_freq, T)  [raw_mag, raw_cos, raw_sin] — pre-activation.

        Returns
        -------
        complex Tensor (B, n_freq, T).
        """
        F = self.n_freq
        raw_mag = raw[:, :F, :]
        raw_cos = raw[:, F:2 * F, :]
        raw_sin = raw[:, 2 * F:, :]

        mag = raw_mag.exp()

        phase_norm = (raw_cos.pow(2) + raw_sin.pow(2)).clamp(min=1e-8).sqrt()
        cos_n = raw_cos / phase_norm
        sin_n = raw_sin / phase_norm

        return torch.complex(mag * cos_n, mag * sin_n)

    # ------------------------------------------------------------------
    # Synthesis and loss
    # ------------------------------------------------------------------

    def forward(
        self,
        raw_batch: AudioBatch,
        target_length: int | None = None,
        wav_lengths: Tensor | None = None,
    ) -> AudioBatch:
        """Reconstruct waveforms from raw decoder predictions.

        Parameters
        ----------
        raw_batch     : AudioBatch  (B, 3 * n_freq, T_frames)  raw predictions.
        target_length : if given, iSTFT is forced to produce exactly this many
                        samples (avoids off-by-one from center padding).
        wav_lengths   : (B,) valid sample counts for the output AudioBatch.
                        If None, estimated from frame lengths.

        Returns
        -------
        AudioBatch  (B, 1, T_samples).
        """
        X = self._raw_to_complex(raw_batch.BCT)      # complex (B, n_freq, T_frames)
        wav = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window,
            center=self.center,
            length=target_length,
        )                                             # (B, T_samples)

        T_out = wav.shape[-1]
        if wav_lengths is None:
            wav_lengths = ((raw_batch.lengths - 1) * self.hop_length).clamp(max=T_out)

        return AudioBatch(
            data=wav.unsqueeze(1),                   # (B, 1, T_samples)
            lengths=wav_lengths,
            sample_rate=self.sample_rate,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss_per_TF(
        self,
        raw: Tensor,
        target: Tensor,
        log_scale: bool,
        norm: str = "l1",
    ) -> Tensor:
        """Per-(freq, time) loss element before frequency averaging.

        Parameters
        ----------
        raw       : (B, 3 * n_freq, T)  raw decoder output [raw_mag, raw_cos, raw_sin].
        target    : (B, 3 * n_freq, T)  extractor output [mag, cos, sin].
        log_scale : if True, compare log-magnitudes and unit-circle phase components
                    separately, equalizing the loss across energy levels.
                    If False, compare directly in the complex STFT domain.
        norm      : ``"l1"`` (absolute value) or ``"l2"`` (squared).

        Returns
        -------
        Tensor of shape (B, n_freq, T).
        """
        penalise = (lambda x: x.abs()) if norm == "l1" else (lambda x: x.pow(2))

        if log_scale:
            F = self.n_freq
            raw_mag = raw[:, :F, :]
            raw_cos = raw[:, F:2 * F, :]
            raw_sin = raw[:, 2 * F:, :]

            mag_target = target[:, :F, :]
            cos_target = target[:, F:2 * F, :]
            sin_target = target[:, 2 * F:, :]

            # raw_mag = log(predicted magnitude) — compare directly with log(target magnitude)
            log_mag_target = mag_target.clamp(min=1e-8).log()
            mag_loss = penalise(raw_mag - log_mag_target)

            # Normalize raw phase to unit circle, then compare with target cos/sin
            phase_norm = (raw_cos.pow(2) + raw_sin.pow(2)).clamp(min=1e-8).sqrt()
            cos_pred = raw_cos / phase_norm
            sin_pred = raw_sin / phase_norm
            phase_loss = penalise(cos_pred - cos_target) + penalise(sin_pred - sin_target)

            return mag_loss + phase_loss                    # (B, n_freq, T)
        else:
            X_pred   = self._raw_to_complex(raw)
            X_target = self._features_to_complex(target)
            diff     = X_pred - X_target
            return penalise(diff.real) + penalise(diff.imag)   # (B, n_freq, T)

    def reconstruction_loss(
        self,
        pred_batch: AudioBatch,
        target_batch: AudioBatch,
        mask: Tensor,
        log_scale: bool = False,
        norm: str = "l1",
    ) -> Tensor:
        """Reconstruction loss in the complex STFT domain, averaged uniformly over frequency bins.

        Parameters
        ----------
        pred_batch   : AudioBatch  (B, 3 * n_freq, T)  raw decoder output.
        target_batch : AudioBatch  (B, 3 * n_freq, T)  extractor output.
        mask         : (B, T)  True = valid frame.
        log_scale    : if True, use log-magnitude + phase loss to equalize
                       the contribution of bins across energy levels.
        norm         : ``"l1"`` or ``"l2"``.

        Returns
        -------
        Scalar loss tensor.
        """
        loss_per_TF = self._compute_loss_per_TF(pred_batch.BCT, target_batch.BCT, log_scale, norm)
        loss_per_T  = loss_per_TF.mean(dim=1)              # (B, T) — avg over freq
        n_valid = mask.float().sum().clamp(min=1)
        return (loss_per_T * mask.float()).sum() / n_valid

    def mel_weighted_reconstruction_loss(
        self,
        pred_batch: AudioBatch,
        target_batch: AudioBatch,
        mask: Tensor,
        log_scale: bool = False,
        norm: str = "l1",
    ) -> Tensor:
        """Reconstruction loss with mel-filterbank frequency weighting.

        Same as ``reconstruction_loss`` but bins are weighted by their cumulative
        mel filterbank response before averaging, giving more importance to
        perceptually salient low frequencies.

        Parameters and return value are identical to ``reconstruction_loss``.
        """
        loss_per_TF = self._compute_loss_per_TF(pred_batch.BCT, target_batch.BCT, log_scale, norm)
        weights     = self._mel_weights.view(1, -1, 1)     # (1, n_freq, 1)
        loss_per_T  = (loss_per_TF * weights).mean(dim=1)  # (B, T) — weighted avg over freq
        n_valid = mask.float().sum().clamp(min=1)
        return (loss_per_T * mask.float()).sum() / n_valid
