"""Metadata DataFrame utilities."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_paths(
    metadata: pd.DataFrame,
    roots: dict[str, str | Path],
    path_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Prepend per-dataset roots to relative path columns.

    Expects a ``corpus`` column that maps each row to an entry in ``roots``.
    All columns listed in ``path_columns`` are resolved; the default is
    ``["signal_path"]``.

    Parameters
    ----------
    metadata     : DataFrame with a ``corpus`` column and one or more
                   relative-path columns.
    roots        : mapping from corpus name to the root directory on the
                   current machine.  Typically loaded from the
                   ``data.dataset_roots`` section of the experiment YAML.
    path_columns : columns to resolve.  Defaults to ``["signal_path"]``.

    Returns
    -------
    A copy of ``metadata`` with the specified columns replaced by absolute paths.

    Raises
    ------
    KeyError
        If any ``corpus`` value in the DataFrame has no entry in ``roots``.
    """
    if path_columns is None:
        path_columns = ["signal_path"]

    missing = set(metadata["corpus"].unique()) - set(roots.keys())
    if missing:
        raise KeyError(
            f"No root defined for corpus: {sorted(missing)}. "
            f"Add them to the 'data.dataset_roots' config section."
        )

    metadata = metadata.copy()
    for col in path_columns:
        metadata[col] = metadata.apply(
            lambda row, c=col: str(Path(roots[row["corpus"]]) / row[c]),
            axis=1,
        )
    return metadata
