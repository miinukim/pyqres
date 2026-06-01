from __future__ import annotations

"""Compressed visibility-projector diagnostics.

The utilities here construct the finite-dimensional objects used by the
operator-isotropy experiments. Given an orthonormalized or raw latent basis
Q_gamma and a readout matrix R, they compute

B_gamma = Q_gamma^* P_vis Q_gamma

and its supported restriction B_gamma_plus after removing exact-null
directions. The eigenvalues of these matrices are sin^2(theta_j) for the
full and supported visibility angles.
"""

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import scipy.linalg as la

from .linalg_utils import null_space, orthonormal_basis_from_columns


def _encode_complex_matrix(matrix: np.ndarray) -> str:
    """Serialize a complex matrix as JSON with separate real/imag parts."""

    arr = np.asarray(matrix, dtype=complex)
    return json.dumps({"real": np.real(arr).tolist(), "imag": np.imag(arr).tolist()})


def _op_norm_hermitian(matrix: np.ndarray) -> float:
    """Operator norm for a Hermitian matrix, computed from eigenvalues."""

    if matrix.size == 0:
        return 0.0
    hermitian = 0.5 * (np.asarray(matrix, dtype=complex) + np.asarray(matrix, dtype=complex).conj().T)
    eigvals = la.eigvalsh(hermitian, check_finite=True)
    return float(np.max(np.abs(np.real_if_close(eigvals)))) if eigvals.size else 0.0


def _max_offdiag_row_sum(matrix: np.ndarray) -> float:
    """Return a simple row-sum measure of off-diagonal leakage."""

    if matrix.size == 0:
        return 0.0
    arr = np.asarray(matrix, dtype=complex)
    offdiag = arr - np.diag(np.diag(arr))
    return float(np.max(np.sum(np.abs(offdiag), axis=1))) if offdiag.size else 0.0


def _orth_projector(basis: np.ndarray, ambient_dim: int) -> np.ndarray:
    """Construct the orthogonal projector onto the span of basis columns."""

    if basis.size == 0 or basis.shape[1] == 0:
        return np.zeros((ambient_dim, ambient_dim), dtype=complex)
    q = orthonormal_basis_from_columns(np.asarray(basis, dtype=complex))
    return q @ q.conj().T


def _visibility_angles_deg(sin2_theta: np.ndarray) -> np.ndarray:
    """Convert sin^2(theta) values into angles in degrees."""

    vals = np.asarray(sin2_theta, dtype=float)
    return np.degrees(np.arcsin(np.sqrt(np.clip(vals, 0.0, 1.0)))) if vals.size else np.array([], dtype=float)


def _theta_star_deg(alpha: float) -> float:
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        return float("nan")
    return float(np.degrees(np.arcsin(np.sqrt(np.clip(alpha, 0.0, 1.0)))))


@dataclass(frozen=True)
class CompressedVisibilityDiagnostics:
    """Structured visibility diagnostics for one latent/readout pair.

    The scalar fields are convenient for CSV tables; the matrix fields are kept
    for downstream analysis that needs the actual compressed projectors.
    """

    ambient_dim: int
    r_visible: int
    alpha_ambient: float
    s_gamma: int
    exact_null_dim: int
    supported_dim: int
    alpha_gamma: float
    alpha_plus: float
    theta_ambient_star_deg: float
    theta_gamma_star_deg: float
    theta_plus_star_deg: float
    delta_iso_gamma: float
    delta_iso_plus: float
    epsilon_diag_gamma: float
    epsilon_diag_plus: float
    epsilon_off_gamma: float
    epsilon_off_plus: float
    theorem_bound_max_violation: float
    b_gamma: np.ndarray
    b_gamma_plus: np.ndarray
    sin2_theta: np.ndarray
    supported_sin2_theta: np.ndarray
    visibility_angle_deg: np.ndarray
    supported_visibility_angle_deg: np.ndarray

    @property
    def s_gamma_plus(self) -> int:
        return self.supported_dim

    @property
    def exact_null_fraction(self) -> float:
        return float(self.exact_null_dim / self.s_gamma) if self.s_gamma > 0 else float("nan")

    @property
    def zero_component_fraction(self) -> float:
        return self.exact_null_fraction

    def as_metrics_dict(self, *, encode_matrices: bool = True) -> dict[str, Any]:
        """Return a flat dict suitable for experiment rows or JSON output."""

        b_gamma: Any = _encode_complex_matrix(self.b_gamma) if encode_matrices else self.b_gamma
        b_gamma_plus: Any = _encode_complex_matrix(self.b_gamma_plus) if encode_matrices else self.b_gamma_plus
        zero_sin2 = self.sin2_theta[: self.exact_null_dim]
        zero_angles = self.visibility_angle_deg[: self.exact_null_dim]
        return {
            "ambient_dim": self.ambient_dim,
            "r_visible": self.r_visible,
            "alpha_ambient": self.alpha_ambient,
            "s_gamma": self.s_gamma,
            "exact_null_dim": self.exact_null_dim,
            "zero_component_dim": self.exact_null_dim,
            "supported_dim": self.supported_dim,
            "s_gamma_plus": self.s_gamma_plus,
            "exact_null_fraction": self.exact_null_fraction,
            "zero_component_fraction": self.zero_component_fraction,
            "alpha_gamma": self.alpha_gamma,
            "alpha_plus": self.alpha_plus,
            "theta_ambient_star_deg": self.theta_ambient_star_deg,
            "theta_gamma_star_deg": self.theta_gamma_star_deg,
            "theta_plus_star_deg": self.theta_plus_star_deg,
            "delta_iso_gamma": self.delta_iso_gamma,
            "delta_iso_plus": self.delta_iso_plus,
            "epsilon_diag_gamma": self.epsilon_diag_gamma,
            "epsilon_diag_plus": self.epsilon_diag_plus,
            "epsilon_off_gamma": self.epsilon_off_gamma,
            "epsilon_off_plus": self.epsilon_off_plus,
            "theorem_bound_max_violation": self.theorem_bound_max_violation,
            "mean_visibility_angle_deg": (
                float(np.mean(self.visibility_angle_deg)) if self.visibility_angle_deg.size else float("nan")
            ),
            "min_visibility_angle_deg": (
                float(np.min(self.visibility_angle_deg)) if self.visibility_angle_deg.size else float("nan")
            ),
            "max_visibility_angle_deg": (
                float(np.max(self.visibility_angle_deg)) if self.visibility_angle_deg.size else float("nan")
            ),
            "mean_supported_visibility_angle_deg": (
                float(np.mean(self.supported_visibility_angle_deg))
                if self.supported_visibility_angle_deg.size
                else float("nan")
            ),
            "min_supported_visibility_angle_deg": (
                float(np.min(self.supported_visibility_angle_deg))
                if self.supported_visibility_angle_deg.size
                else float("nan")
            ),
            "max_supported_visibility_angle_deg": (
                float(np.max(self.supported_visibility_angle_deg))
                if self.supported_visibility_angle_deg.size
                else float("nan")
            ),
            "var_sin2_theta": float(np.var(self.sin2_theta)) if self.sin2_theta.size else 0.0,
            "var_supported_sin2_theta": (
                float(np.var(self.supported_sin2_theta)) if self.supported_sin2_theta.size else 0.0
            ),
            "sin2_theta": json.dumps([float(x) for x in self.sin2_theta]),
            "zero_component_sin2_theta": json.dumps([float(x) for x in zero_sin2]),
            "supported_sin2_theta": json.dumps([float(x) for x in self.supported_sin2_theta]),
            "visibility_angle_deg": json.dumps([float(x) for x in self.visibility_angle_deg]),
            "zero_component_visibility_angle_deg": json.dumps([float(x) for x in zero_angles]),
            "supported_visibility_angle_deg": json.dumps([float(x) for x in self.supported_visibility_angle_deg]),
            "B_gamma_json": b_gamma,
            "B_gamma_plus_json": b_gamma_plus,
            "B_gamma_eigenvalues": json.dumps([float(x) for x in self.sin2_theta]),
            "B_gamma_plus_eigenvalues": json.dumps([float(x) for x in self.supported_sin2_theta]),
        }


def compressed_visibility_diagnostics(
    latent_basis: np.ndarray,
    readout_matrix: np.ndarray,
    *,
    tol: float = 1e-10,
) -> CompressedVisibilityDiagnostics:
    """Construct B_gamma, B_gamma_plus, and their isotropy metrics."""

    q_gamma = orthonormal_basis_from_columns(np.asarray(latent_basis, dtype=complex), tol=tol)
    s_gamma = int(q_gamma.shape[1])
    ambient_dim = int(readout_matrix.shape[1])
    readout_nullspace = null_space(readout_matrix, tol=tol)
    # Visible directions are the orthogonal complement of the readout nullspace.
    p_null = _orth_projector(readout_nullspace, ambient_dim)
    p_vis = np.eye(ambient_dim, dtype=complex) - p_null
    p_vis = 0.5 * (p_vis + p_vis.conj().T)
    r_visible = int(ambient_dim - readout_nullspace.shape[1])
    alpha_ambient = float(r_visible / ambient_dim) if ambient_dim > 0 else float("nan")

    if s_gamma == 0:
        empty = np.zeros((0, 0), dtype=complex)
        return CompressedVisibilityDiagnostics(
            ambient_dim=ambient_dim,
            r_visible=r_visible,
            alpha_ambient=alpha_ambient,
            s_gamma=0,
            exact_null_dim=0,
            supported_dim=0,
            alpha_gamma=float("nan"),
            alpha_plus=float("nan"),
            theta_ambient_star_deg=_theta_star_deg(alpha_ambient),
            theta_gamma_star_deg=float("nan"),
            theta_plus_star_deg=float("nan"),
            delta_iso_gamma=float("nan"),
            delta_iso_plus=float("nan"),
            epsilon_diag_gamma=float("nan"),
            epsilon_diag_plus=float("nan"),
            epsilon_off_gamma=float("nan"),
            epsilon_off_plus=float("nan"),
            theorem_bound_max_violation=float("nan"),
            b_gamma=empty,
            b_gamma_plus=empty,
            sin2_theta=np.array([], dtype=float),
            supported_sin2_theta=np.array([], dtype=float),
            visibility_angle_deg=np.array([], dtype=float),
            supported_visibility_angle_deg=np.array([], dtype=float),
        )

    b_gamma = q_gamma.conj().T @ p_vis @ q_gamma
    b_gamma = 0.5 * (b_gamma + b_gamma.conj().T)
    # Eigenvalues of B_gamma are sin^2(theta_j), including exact-null directions.
    evals, evecs = la.eigh(b_gamma, check_finite=True)
    evals = np.clip(np.real_if_close(evals), 0.0, 1.0)
    visibility_angles = _visibility_angles_deg(evals)
    alpha_gamma = float(np.real_if_close(np.trace(b_gamma)) / s_gamma)
    delta_iso_gamma = _op_norm_hermitian(b_gamma - alpha_gamma * np.eye(s_gamma, dtype=complex))
    diag_gamma = np.real_if_close(np.diag(b_gamma))
    epsilon_diag_gamma = float(np.max(np.abs(diag_gamma - alpha_gamma))) if diag_gamma.size else 0.0
    epsilon_off_gamma = _max_offdiag_row_sum(b_gamma)

    supported_mask = evals > tol
    supported_dim = int(np.sum(supported_mask))
    exact_null_dim = int(s_gamma - supported_dim)
    # Removing exact-null directions separates "the readout misses a subspace
    # entirely" from anisotropy inside the directions it does see.
    if supported_dim == 0:
        empty = np.zeros((0, 0), dtype=complex)
        return CompressedVisibilityDiagnostics(
            ambient_dim=ambient_dim,
            r_visible=r_visible,
            alpha_ambient=alpha_ambient,
            s_gamma=s_gamma,
            exact_null_dim=exact_null_dim,
            supported_dim=0,
            alpha_gamma=alpha_gamma,
            alpha_plus=float("nan"),
            theta_ambient_star_deg=_theta_star_deg(alpha_ambient),
            theta_gamma_star_deg=_theta_star_deg(alpha_gamma),
            theta_plus_star_deg=float("nan"),
            delta_iso_gamma=float(delta_iso_gamma),
            delta_iso_plus=float("nan"),
            epsilon_diag_gamma=epsilon_diag_gamma,
            epsilon_diag_plus=float("nan"),
            epsilon_off_gamma=epsilon_off_gamma,
            epsilon_off_plus=float("nan"),
            theorem_bound_max_violation=float("nan"),
            b_gamma=b_gamma,
            b_gamma_plus=empty,
            sin2_theta=evals,
            supported_sin2_theta=np.array([], dtype=float),
            visibility_angle_deg=visibility_angles,
            supported_visibility_angle_deg=np.array([], dtype=float),
        )

    v0 = evecs[:, ~supported_mask]
    p_plus_coeff = np.eye(s_gamma, dtype=complex) - v0 @ v0.conj().T if v0.size else np.eye(s_gamma, dtype=complex)
    q_plus = orthonormal_basis_from_columns(q_gamma @ p_plus_coeff, tol=tol)
    supported_dim = int(q_plus.shape[1])
    if supported_dim == 0:
        empty = np.zeros((0, 0), dtype=complex)
        return CompressedVisibilityDiagnostics(
            ambient_dim=ambient_dim,
            r_visible=r_visible,
            alpha_ambient=alpha_ambient,
            s_gamma=s_gamma,
            exact_null_dim=s_gamma,
            supported_dim=0,
            alpha_gamma=alpha_gamma,
            alpha_plus=float("nan"),
            theta_ambient_star_deg=_theta_star_deg(alpha_ambient),
            theta_gamma_star_deg=_theta_star_deg(alpha_gamma),
            theta_plus_star_deg=float("nan"),
            delta_iso_gamma=float(delta_iso_gamma),
            delta_iso_plus=float("nan"),
            epsilon_diag_gamma=epsilon_diag_gamma,
            epsilon_diag_plus=float("nan"),
            epsilon_off_gamma=epsilon_off_gamma,
            epsilon_off_plus=float("nan"),
            theorem_bound_max_violation=float("nan"),
            b_gamma=b_gamma,
            b_gamma_plus=empty,
            sin2_theta=evals,
            supported_sin2_theta=np.array([], dtype=float),
            visibility_angle_deg=visibility_angles,
            supported_visibility_angle_deg=np.array([], dtype=float),
        )

    b_plus = q_plus.conj().T @ p_vis @ q_plus
    b_plus = 0.5 * (b_plus + b_plus.conj().T)
    plus_evals = np.clip(np.real_if_close(la.eigvalsh(b_plus, check_finite=True)), 0.0, 1.0)
    plus_angles = _visibility_angles_deg(plus_evals)
    alpha_plus = float(np.real_if_close(np.trace(b_plus)) / supported_dim)
    delta_iso_plus = _op_norm_hermitian(b_plus - alpha_plus * np.eye(supported_dim, dtype=complex))
    diag_plus = np.real_if_close(np.diag(b_plus))
    epsilon_diag_plus = float(np.max(np.abs(diag_plus - alpha_plus))) if diag_plus.size else 0.0
    epsilon_off_plus = _max_offdiag_row_sum(b_plus)
    max_violation = float(np.max(np.abs(plus_evals - alpha_plus)) - delta_iso_plus) if plus_evals.size else 0.0

    return CompressedVisibilityDiagnostics(
        ambient_dim=ambient_dim,
        r_visible=r_visible,
        alpha_ambient=alpha_ambient,
        s_gamma=s_gamma,
        exact_null_dim=int(s_gamma - supported_dim),
        supported_dim=supported_dim,
        alpha_gamma=alpha_gamma,
        alpha_plus=alpha_plus,
        theta_ambient_star_deg=_theta_star_deg(alpha_ambient),
        theta_gamma_star_deg=_theta_star_deg(alpha_gamma),
        theta_plus_star_deg=_theta_star_deg(alpha_plus),
        delta_iso_gamma=float(delta_iso_gamma),
        delta_iso_plus=float(delta_iso_plus),
        epsilon_diag_gamma=epsilon_diag_gamma,
        epsilon_diag_plus=epsilon_diag_plus,
        epsilon_off_gamma=epsilon_off_gamma,
        epsilon_off_plus=epsilon_off_plus,
        theorem_bound_max_violation=max_violation,
        b_gamma=b_gamma,
        b_gamma_plus=b_plus,
        sin2_theta=evals,
        supported_sin2_theta=plus_evals,
        visibility_angle_deg=visibility_angles,
        supported_visibility_angle_deg=plus_angles,
    )


def compressed_visibility_metrics(
    latent_basis: np.ndarray,
    readout_matrix: np.ndarray,
    *,
    tol: float = 1e-10,
    encode_matrices: bool = True,
) -> dict[str, Any]:
    """Return flat CSV-friendly metrics for compressed visibility projectors."""

    return compressed_visibility_diagnostics(
        latent_basis,
        readout_matrix,
        tol=tol,
    ).as_metrics_dict(encode_matrices=encode_matrices)


__all__ = [
    "CompressedVisibilityDiagnostics",
    "compressed_visibility_diagnostics",
    "compressed_visibility_metrics",
]
