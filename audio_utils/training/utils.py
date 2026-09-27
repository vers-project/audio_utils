"""Shared training utilities."""
from __future__ import annotations

import torch
import torch.nn as nn


def make_trainer_kwargs(
    find_unused_parameters: bool = False,
    static_graph: bool = False,
) -> dict:
    """Return Lightning Trainer keyword arguments for hardware-adaptive training.

    Detects the number of available GPUs at runtime and returns the appropriate
    ``accelerator``, ``devices``, and ``strategy`` combination:

    - **Multiple GPUs** → DDP across all visible GPUs.
    - **Single GPU**    → single-GPU mode, no strategy overhead.
    - **CPU only**      → CPU training, no strategy.

    Parameters
    ----------
    find_unused_parameters : passed to ``DDPStrategy``.  Set to ``True`` only
        if parameters genuinely do not participate in every backward and
        ``static_graph`` is not suitable.
    static_graph : passed to ``DDPStrategy``.  Set to ``True`` for GAN-style
        training with manual optimization where only a subset of parameters
        receives gradients per backward (e.g. codec params in the G step,
        discriminator params in the D step).  DDP skips per-backward graph
        traversal and instead assumes the set of active parameters is
        consistent across iterations — zero runtime overhead compared to
        ``find_unused_parameters=True``.

    Usage
    -----
    ::

        # Standard reconstruction training
        trainer = L.Trainer(**make_trainer_kwargs(), max_epochs=100)

        # GAN training with toggle_optimizer
        trainer = L.Trainer(**make_trainer_kwargs(static_graph=True), max_epochs=100)
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        from lightning.pytorch.strategies import DDPStrategy
        return {
            "accelerator": "gpu",
            "devices": n_gpus,
            "strategy": DDPStrategy(
                find_unused_parameters=find_unused_parameters,
                static_graph=static_graph,
            ),
        }
    if n_gpus == 1:
        return {"accelerator": "gpu", "devices": 1}
    return {"accelerator": "cpu"}


def configure_matmul_precision(precision: str = "medium") -> None:
    """Set float32 matmul precision on GPUs that have Tensor Cores (compute ≥ 7.0).

    Volta (7.0) and newer architectures support TF32, which trades a small
    amount of float32 precision for significantly faster matmuls.  This is a
    no-op on CPU-only machines or older GPUs.

    Call once at the top of each training script's ``main()`` before the
    ``Trainer`` is constructed.

    Parameters
    ----------
    precision : "medium" (TF32, recommended) or "high" (more precise TF32).
    """
    if not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major >= 7:
        torch.set_float32_matmul_precision(precision)


def apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Recursively apply spectral normalisation to all Linear and Conv layers.

    Modifies the module in-place and returns it for convenience.
    Useful for stabilising GAN discriminator training.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            setattr(module, name, nn.utils.spectral_norm(child))
        else:
            apply_spectral_norm(child)
    return module
