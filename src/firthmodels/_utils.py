import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Protocol, Sequence


@dataclass
class FirthResult:
    """Output from Firth-penalized optimization"""

    beta: NDArray[np.float64]  # (n_features,) fitted coefficients
    loglik: float  # fitted log-likelihood
    fisher_info: NDArray[
        np.float64
    ]  # (n_features, n_features) Fisher information matrix
    n_iter: int  # number of iterations
    converged: bool  # whether optimization converged


class IterationQuantities(Protocol):
    """Quantities computed at each iteration of optimization"""

    loglik: float
    modified_score: NDArray[np.float64]
    fisher_info: NDArray[np.float64]


def resolve_feature_indices(
    features: int | str | Sequence[int | str] | None,
    *,
    n_params: int,
    feature_names_in: NDArray[np.str_] | None = None,
    intercept_idx: int | None = None,
) -> list[int]:
    """Convert feature names and/or indices to list of parameter indices."""
    if features is None:
        return list(range(n_params))

    features_seq = (
        [features] if isinstance(features, (int, np.integer, str)) else features
    )

    feature_names_map = (
        {name: i for i, name in enumerate(feature_names_in)}
        if feature_names_in is not None
        else None
    )

    indices = []
    for feat in features_seq:
        if isinstance(feat, str):
            if feat == "intercept":
                if intercept_idx is None:
                    raise ValueError("Model has no intercept")
                indices.append(intercept_idx)
            elif feature_names_map is None:
                raise ValueError(
                    "No feature names available. Pass a DataFrame to fit(), or use "
                    "integer indices."
                )
            else:
                try:
                    indices.append(feature_names_map[feat])
                except KeyError:
                    raise KeyError(f"Unknown feature: '{feat}'") from None
        elif isinstance(feat, (int, np.integer)):
            indices.append(int(feat))
        else:
            raise TypeError(
                f"Elements of `features` must be int or str, got {type(feat)}"
            )
    return indices
