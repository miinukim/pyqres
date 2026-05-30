from __future__ import annotations

"""Volterra analysis pipelines.

This module contains both:

1. the original dense PTM-to-Volterra pipeline
2. an exact reduced observable-side pipeline based on ``(P, L)``-truncated reduced sector

The dense path remains useful for small systems and regression checks. The
reduced path avoids explicit dense PTM construction by generating only the
observable-relevant operator sector under admissible zero-input drift and
input-insertion words.
"""

from dataclasses import dataclass
from math import factorial
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np
import scipy.linalg as la

from .linalg_utils import (
    checked_matmul,
    derivative_from_samples,
    effective_rank_psd,
    ensure_finite,
    hs_inner_product,
    matrix_rank,
    null_space,
    orthogonalize_operator,
    orthonormal_basis_from_columns,
    positive_matrix_pseudoinverse_sqrt,
    principal_angles,
    ridge_effective_dimension_psd,
)

FeatureLabel = Tuple[int, ...]
# Example convention used in comments below:
# - ``n_memory = 2`` so the traceless PTM sector has dimension ``4^2 - 1 = 15``
# - ``observables = [Z0, Z1]`` so visible matrices have ``J = 2`` rows
# - ``P = 2``, ``L = 1`` for the reduced/dense truncation examples
# - dense labels look like ``(1, 0)`` or ``(2, 0)`` in the polynomial-history basis
# - reduced labels look like ``(2, 0, 1)`` meaning:
#   seed observable 2 -> one drift step -> one first-order insertion


class ReservoirModelProtocol(Protocol):
    """Minimal interface required by the Volterra analysis code."""

    dim_memory: int
    memory_basis: Sequence[np.ndarray]

    def channel(self, u: float, op_memory: np.ndarray) -> np.ndarray: ...

    def ptm(self, u: float) -> np.ndarray: ...

    def channel_adjoint(self, u: float, observable_memory: np.ndarray) -> np.ndarray: ...

    def channel_derivative_adjoint(
        self,
        order: int,
        observable_memory: np.ndarray,
        u0: float = 0.0,
        fd_step: float = 5e-3,
        stencil_radius: int | None = None,
    ) -> np.ndarray: ...

    def fixed_point(self, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray: ...

    def readout_matrix(self, observables: Sequence[np.ndarray]) -> np.ndarray: ...

    def default_memory_observables(
        self,
        preset: str = "z",
        custom_specs: Sequence[str] | None = None,
    ) -> List[np.ndarray]: ...


@dataclass
class VolterraResult:
    """Container for the numerical diagnostics produced by one analysis run."""

    # ``monomials`` is used generically for both analysis paths:
    # - dense PTM path: polynomial-history labels for the truncated Volterra family
    # - reduced path: one admissible reduced-sector word per retained basis vector
    monomials: List[FeatureLabel]
    # ``latent_kernel_matrix`` is the raw spanning family before orthonormalization.
    # Its columns are PTM traceless coordinates in the dense path and flattened
    # operator columns in the reduced observable-side path.
    latent_kernel_matrix: np.ndarray
    # ``latent_basis_matrix`` is an orthonormal basis for the span that is actually
    # used for principal-angle and Gram-geometry diagnostics.
    latent_basis_matrix: np.ndarray
    # ``restricted_measurement_matrix`` is the ``O_{P,L}`` object: the
    # readout map restricted to an orthonormal basis of the relevant span.
    restricted_measurement_matrix: np.ndarray
    # ``visible_coefficient_matrix`` is the kernel-side matrix used for hard OVD.
    # In the reduced analyzer we identify it with the restricted matrix because the
    # reduced sector is already constructed directly on the observable side.
    visible_coefficient_matrix: np.ndarray
    visible_gram_matrix: np.ndarray
    # ``covariance_matrix`` is the noise model used for whitening. By default this
    # is isotropic with scale ``tau(N, delta)^2``, but callers may override it.
    covariance_matrix: np.ndarray
    whitened_restricted_measurement_matrix: np.ndarray
    whitened_visible_coefficient_matrix: np.ndarray
    whitened_visible_gram_matrix: np.ndarray
    singular_values: np.ndarray
    restricted_singular_values: np.ndarray
    whitened_singular_values: np.ndarray
    visible_gram_eigenvalues: np.ndarray
    whitened_gram_eigenvalues: np.ndarray
    vvr: int
    ovd: int
    whitened_ovd: int
    soft_ovd: float
    latent_dim: int
    visible_effective_rank: float
    whitened_effective_rank: float
    principal_angles_rad: np.ndarray
    principal_angles_deg: np.ndarray
    noise_threshold: float


class PTMAffineExpansion:
    """Numerically differentiate the PTM around a reference input."""

    def __init__(
        self,
        model: ReservoirModelProtocol,
        max_order: int,
        fd_step: float = 5e-3,
        stencil_radius: int | None = None,
        expansion_point: float = 0.0,
    ):
        self.model = model
        self.max_order = max_order
        self.fd_step = fd_step
        self.stencil_radius = stencil_radius if stencil_radius is not None else max(2, max_order + 1)
        self.expansion_point = float(expansion_point)
        self.points = list(range(-self.stencil_radius, self.stencil_radius + 1))
        self.T_samples: Dict[int, np.ndarray] = {}
        self.A_derivs: Dict[int, np.ndarray] = {}
        self.b_derivs: Dict[int, np.ndarray] = {}
        self.rho_bar: np.ndarray | None = None
        self.xbar_tr: np.ndarray | None = None
        self._compute()

    def _compute(self) -> None:
        T0 = self.model.ptm(self.expansion_point)
        # Example: for ``n_memory = 2``, ``T0.shape == (16, 16)`` and the
        # traceless block ``T0[1:, 1:]`` has shape ``(15, 15)``.
        ensure_finite("PTM at expansion point", T0)
        rho = np.eye(self.model.dim_memory, dtype=complex) / self.model.dim_memory
        # Example initial state for ``n_memory = 2``:
        # ``rho = I_4 / 4 = diag(0.25, 0.25, 0.25, 0.25)``.
        # Iterate to find the fixed point
        for _ in range(10000):
            new_rho = self.model.channel(self.expansion_point, rho)
            new_rho = 0.5 * (new_rho + new_rho.conj().T)
            ensure_finite("fixed-point iterate", new_rho)
            tr = np.trace(new_rho)
            if abs(tr) > 1e-15:
                new_rho /= tr
            ensure_finite("normalized fixed-point iterate", new_rho)
            if np.linalg.norm(new_rho - rho, ord="fro") < 1e-12:
                rho = new_rho
                break
            rho = new_rho
        self.rho_bar = rho
        ensure_finite("fixed point rho_bar", self.rho_bar)

        basis = self.model.memory_basis
        dim = self.model.dim_memory
        xbar = []
        for idx, pauli in enumerate(basis):
            product = checked_matmul(f"basis[{idx}] @ rho_bar", pauli, self.rho_bar)
            xbar.append(np.trace(product) / dim)
        self.xbar_tr = ensure_finite("xbar_tr", np.array(xbar, dtype=complex)[1:])

        A_samples = []
        b_samples = []
        for point in self.points:
            u = self.expansion_point + point * self.fd_step
            # Example with ``fd_step = 5e-3`` and ``points = [-3, ..., 3]``:
            # sampled inputs are ``-0.015, -0.010, ..., 0.015`` around ``u0 = 0``.
            T = self.model.ptm(u)
            ensure_finite(f"PTM at u={u}", T)
            self.T_samples[point] = T
            A_u = T[1:, 1:]
            c_u = T[1:, 0]
            Ax = checked_matmul(f"A_u @ xbar_tr at u={u}", A_u, self.xbar_tr)
            b_u = ensure_finite(f"b_u at u={u}", c_u + Ax - self.xbar_tr)
            # Example shapes for ``n_memory = 2``:
            # ``A_u.shape == (15, 15)``, ``c_u.shape == (15,)``, ``b_u.shape == (15,)``.
            A_samples.append(ensure_finite(f"A_u at u={u}", A_u))
            b_samples.append(b_u)

        self.A_derivs[0] = ensure_finite("A_derivs[0]", T0[1:, 1:])
        self.b_derivs[0] = np.zeros_like(self.xbar_tr)
        for order in range(1, self.max_order + 1):
            self.A_derivs[order] = ensure_finite(
                f"A_derivs[{order}]",
                derivative_from_samples(A_samples, self.fd_step, order, self.points),
            )
            self.b_derivs[order] = ensure_finite(
                f"b_derivs[{order}]",
                derivative_from_samples(b_samples, self.fd_step, order, self.points),
            )

    @property
    def A0(self) -> np.ndarray:
        return self.A_derivs[0]

    def Ak(self, k: int) -> np.ndarray:
        return self.A_derivs[k]

    def bk(self, k: int) -> np.ndarray:
        return self.b_derivs[k]


class TruncatedVolterraGenerator:
    """Generate a finite polynomial-history family from the affine expansion."""

    def __init__(self, expansion: PTMAffineExpansion, max_order: int, lag_horizon: int):
        self.expansion = expansion
        self.max_order = max_order
        self.lag_horizon = lag_horizon
        self.dim_tr = expansion.A0.shape[0]

    def generate(self, tol: float = 1e-10) -> Tuple[List[FeatureLabel], np.ndarray]:
        # Initialize with zero monomial
        zero = (0,) * (self.lag_horizon + 1)
        # Example with ``L = 2``: ``zero == (0, 0, 0)``.
        state: Dict[FeatureLabel, np.ndarray] = {zero: np.zeros(self.dim_tr, dtype=complex)}

        for _ in range(self.lag_horizon + 1):
            shifted: Dict[FeatureLabel, np.ndarray] = {}
            # Shift old monomials one lag deeper
            for monomial, coeff in state.items():
                new_monomial = (0,) + monomial[:-1]
                # Example: ``(1, 0, 0)`` becomes ``(0, 1, 0)`` after one lag shift.
                shifted[new_monomial] = shifted.get(new_monomial, 0) + coeff

            new_state: Dict[FeatureLabel, np.ndarray] = {}
            for monomial, coeff in shifted.items():
                current_degree = sum(monomial)
                for k in range(0, self.max_order - current_degree + 1):
                    beta = list(monomial)
                    beta[0] += k
                    beta_t = tuple(beta)
                    if k == 0:
                        contrib = self.expansion.A0 @ coeff
                    else:
                        contrib = (self.expansion.Ak(k) @ coeff) / factorial(k)
                    # Example: if ``monomial == (0, 1, 0)`` and ``k == 2``,
                    # then ``beta_t == (2, 1, 0)`` and the contribution uses
                    # ``A_2 / 2!`` on the old coefficient vector.
                    new_state[beta_t] = new_state.get(beta_t, 0) + contrib

            for k in range(1, self.max_order + 1):
                monomial = (k,) + (0,) * self.lag_horizon
                contrib = self.expansion.bk(k) / factorial(k)
                # Example with ``L = 2``:
                # ``k = 1`` seeds ``(1, 0, 0)``, ``k = 2`` seeds ``(2, 0, 0)``.
                new_state[monomial] = new_state.get(monomial, 0) + contrib

            pruned: Dict[FeatureLabel, np.ndarray] = {}
            for monomial, coeff in new_state.items():
                if sum(monomial) <= self.max_order and np.linalg.norm(coeff) > tol:
                    pruned[monomial] = coeff
            state = pruned

        monomials = sorted(state.keys(), key=lambda label: (sum(label), label))
        active = [label for label in monomials if sum(label) > 0 and np.linalg.norm(state[label]) > tol]
        columns = [state[label] for label in active]
        # Example: ``active`` might look like
        # ``[(1, 0, 0), (0, 1, 0), (2, 0, 0), (1, 1, 0)]`` and
        # ``K.shape == (15, 4)`` for the toy ``n_memory = 2`` case.
        K = np.column_stack(columns) if columns else np.zeros((self.dim_tr, 0), dtype=complex)
        return active, K


@dataclass(frozen=True)
class _ReducedWordState:
    operator: np.ndarray
    total_order: int
    total_drift: int
    seed_index: int
    word: FeatureLabel


def _noise_threshold(n_shots: int, delta: float, noise_scale: float) -> float:
    # Example: ``n_shots = 2000``, ``delta = 0.05``, ``noise_scale = 1`` gives
    # ``tau ~= sqrt(log(20) / 2000) ~= 0.0387``.
    return float(noise_scale * np.sqrt(np.log(max(2.0, 1.0 / delta)) / max(1, n_shots)))


def _empty_result(
    monomials: List[FeatureLabel],
    latent_kernel_matrix: np.ndarray,
    covariance_matrix: np.ndarray,
    noise_threshold: float,
) -> VolterraResult:
    # Keep the zero-feature case explicit so callers always receive a fully shaped
    # result object and do not need to special-case empty matrices downstream.
    n_visible = covariance_matrix.shape[0]
    latent_basis = np.zeros((latent_kernel_matrix.shape[0], 0), dtype=complex)
    restricted = np.zeros((n_visible, 0), dtype=complex)
    gram = np.zeros((0, 0), dtype=complex)
    # Example empty shapes:
    # ``latent_kernel_matrix.shape == (15, 0)``, ``restricted.shape == (2, 0)``,
    # ``gram.shape == (0, 0)``.
    return VolterraResult(
        monomials=monomials,
        latent_kernel_matrix=latent_kernel_matrix,
        latent_basis_matrix=latent_basis,
        restricted_measurement_matrix=restricted,
        visible_coefficient_matrix=np.zeros((n_visible, latent_kernel_matrix.shape[1]), dtype=complex),
        visible_gram_matrix=gram,
        covariance_matrix=np.asarray(covariance_matrix, dtype=complex),
        whitened_restricted_measurement_matrix=restricted,
        whitened_visible_coefficient_matrix=np.zeros((n_visible, latent_kernel_matrix.shape[1]), dtype=complex),
        whitened_visible_gram_matrix=gram,
        singular_values=np.array([], dtype=float),
        restricted_singular_values=np.array([], dtype=float),
        whitened_singular_values=np.array([], dtype=float),
        visible_gram_eigenvalues=np.array([], dtype=float),
        whitened_gram_eigenvalues=np.array([], dtype=float),
        vvr=0,
        ovd=0,
        whitened_ovd=0,
        soft_ovd=0.0,
        latent_dim=0,
        visible_effective_rank=0.0,
        whitened_effective_rank=0.0,
        principal_angles_rad=np.array([], dtype=float),
        principal_angles_deg=np.array([], dtype=float),
        noise_threshold=float(noise_threshold),
    )


def _coerce_covariance_matrix(
    covariance_model: float | np.ndarray | None,
    n_visible: int,
    default_noise_threshold: float,
) -> np.ndarray:
    # Whitening is defined for a positive definite covariance model on the visible
    # coordinates. The helper accepts a few convenient user-facing forms:
    # ``None`` -> isotropic proxy, scalar -> scalar multiple of identity,
    # vector -> diagonal covariance, matrix -> full covariance model.
    if covariance_model is None:
        # Example default with ``J = 2`` and ``tau = 0.04``:
        # returns ``diag([0.0016, 0.0016])``.
        return (default_noise_threshold**2) * np.eye(n_visible, dtype=complex)
    if np.isscalar(covariance_model):
        variance = float(covariance_model)
        if variance <= 0.0:
            raise ValueError(f"Covariance scalar must be positive, got {variance}.")
        return variance * np.eye(n_visible, dtype=complex)

    cov = np.asarray(covariance_model, dtype=complex)
    if cov.ndim == 1:
        # Example: input ``[0.01, 0.02]`` becomes
        # ``[[0.01, 0.00], [0.00, 0.02]]``.
        if cov.shape[0] != n_visible:
            raise ValueError(
                f"Covariance diagonal has length {cov.shape[0]}, expected {n_visible} visible coordinates."
            )
        if np.any(np.real_if_close(cov) <= 0.0):
            raise ValueError("Covariance diagonal entries must be positive.")
        return np.diag(cov)
    if cov.ndim != 2 or cov.shape != (n_visible, n_visible):
        raise ValueError(
            f"Covariance matrix must have shape {(n_visible, n_visible)}, got {cov.shape}."
        )
    return 0.5 * (cov + cov.conj().T)


def _ambient_readout_matrix(observables: Sequence[np.ndarray]) -> np.ndarray:
    if not observables:
        return np.zeros((0, 0), dtype=complex)
    # Flatten operators in Hilbert-Schmidt geometry so the nullspace calculation
    # can be performed with ordinary linear algebra in the ambient operator space.
    # Example with two 4x4 observables ``Z0`` and ``Z1``:
    # returns a matrix with shape ``(2, 16)``.
    return ensure_finite(
        "ambient readout matrix",
        np.vstack([np.asarray(observable, dtype=complex).reshape(1, -1).conj() for observable in observables]),
    )


def _finalize_result(
    *,
    monomials: List[FeatureLabel],
    latent_kernel_matrix: np.ndarray,
    latent_basis_matrix: np.ndarray,
    restricted_measurement_matrix: np.ndarray,
    visible_coefficient_matrix: np.ndarray,
    readout_nullspace_matrix: np.ndarray,
    covariance_matrix: np.ndarray,
    noise_threshold: float,
    algebraic_tol: float,
    whitened_operational_threshold: float = 2.0,
) -> VolterraResult:
    # This helper centralizes the visible-side diagnostics so the dense PTM path
    # and the reduced observable-side path are evaluated through exactly the same
    # post-processing layer once they supply their span/basis matrices.
    latent_basis_matrix = ensure_finite("latent basis matrix", latent_basis_matrix)
    restricted_measurement_matrix = ensure_finite("restricted measurement matrix", restricted_measurement_matrix)
    visible_coefficient_matrix = ensure_finite("visible coefficient matrix", visible_coefficient_matrix)
    covariance_matrix = ensure_finite("covariance matrix", covariance_matrix)
    # Example shapes in one small run:
    # ``latent_basis_matrix.shape == (15, 6)``
    # ``restricted_measurement_matrix.shape == (2, 6)``
    # ``visible_coefficient_matrix.shape == (2, 9)``.

    # Whitening moves the visible coordinates into the statistically natural metric
    # defined by ``Sigma^{-1}``. In the isotropic default this simply rescales all
    # visible directions by ``1 / tau``.
    whitening = positive_matrix_pseudoinverse_sqrt(covariance_matrix, tol=algebraic_tol)
    whitened_restricted = ensure_finite(
        "whitened restricted measurement matrix",
        whitening @ restricted_measurement_matrix,
    )
    whitened_visible_coefficient = ensure_finite(
        "whitened visible coefficient matrix",
        whitening @ visible_coefficient_matrix,
    )
    # Example with ``Sigma = diag([0.01, 0.04])``:
    # ``whitening = diag([10, 5])`` and each row of the visible matrices is
    # rescaled by its inverse noise standard deviation.

    # ``G_v = O^* O`` and ``G_tilde = O_tilde^* O_tilde`` are the Euclidean and
    # noise-whitened Gram operators used for effective-rank and soft-OVD metrics.
    visible_gram = ensure_finite(
        "visible Gram matrix",
        restricted_measurement_matrix.conj().T @ restricted_measurement_matrix,
    )
    whitened_visible_gram = ensure_finite(
        "whitened visible Gram matrix",
        whitened_restricted.conj().T @ whitened_restricted,
    )
    # Example: if ``restricted`` has shape ``(2, 6)``, both Gram matrices have
    # shape ``(6, 6)`` and act on latent-basis coordinates, not visible rows.

    # Hard OVD is defined on the singular spectrum of the kernel-side visible
    # matrix, whereas the restricted matrix drives the geometric diagnostics.
    singular_values = ensure_finite(
        "visible singular values",
        la.svdvals(visible_coefficient_matrix) if visible_coefficient_matrix.size else np.array([], dtype=float),
    )
    restricted_singular_values = ensure_finite(
        "restricted singular values",
        la.svdvals(restricted_measurement_matrix) if restricted_measurement_matrix.size else np.array([], dtype=float),
    )
    whitened_singular_values = ensure_finite(
        "whitened visible singular values",
        la.svdvals(whitened_visible_coefficient) if whitened_visible_coefficient.size else np.array([], dtype=float),
    )
    # Example singular spectra:
    # ``singular_values = [0.12, 0.03]``
    # ``whitened_singular_values = [3.1, 0.8]`` after row whitening.

    visible_gram_eigenvalues = np.clip(
        np.real_if_close(la.eigvalsh(0.5 * (visible_gram + visible_gram.conj().T), check_finite=True)),
        0.0,
        None,
    )
    whitened_gram_eigenvalues = np.clip(
        np.real_if_close(la.eigvalsh(0.5 * (whitened_visible_gram + whitened_visible_gram.conj().T), check_finite=True)),
        0.0,
        None,
    )

    # ``VVR`` is defined algebraically on the visible image. In exact arithmetic
    # the kernel-side matrix ``C`` and the restricted matrix ``O`` have the same
    # rank, but the restricted matrix is numerically better conditioned because it
    # is built after orthonormalizing the latent span. Use it as the canonical
    # rank source so geometry-side quantities such as soft OVD cannot exceed VVR
    # just because the raw spanning family in ``C`` is ill-conditioned.
    vvr = matrix_rank(restricted_measurement_matrix, tol=algebraic_tol)
    ovd = int(np.sum(np.real_if_close(singular_values) > 2.0 * noise_threshold))
    whitened_ovd = int(np.sum(np.real_if_close(whitened_singular_values) > whitened_operational_threshold))
    soft_ovd = min(
        ridge_effective_dimension_psd(whitened_visible_gram, tol=algebraic_tol),
        float(vvr),
    )
    angles = ensure_finite(
        "principal angles",
        principal_angles(latent_basis_matrix, readout_nullspace_matrix),
    )
    # Example: if ``tau = 0.04`` and ``singular_values = [0.12, 0.03]``,
    # then ``ovd = 1`` because only ``0.12 > 2 * 0.04``.

    return VolterraResult(
        monomials=monomials,
        latent_kernel_matrix=np.real_if_close(latent_kernel_matrix),
        latent_basis_matrix=np.real_if_close(latent_basis_matrix),
        restricted_measurement_matrix=np.real_if_close(restricted_measurement_matrix),
        visible_coefficient_matrix=np.real_if_close(visible_coefficient_matrix),
        visible_gram_matrix=np.real_if_close(visible_gram),
        covariance_matrix=np.real_if_close(covariance_matrix),
        whitened_restricted_measurement_matrix=np.real_if_close(whitened_restricted),
        whitened_visible_coefficient_matrix=np.real_if_close(whitened_visible_coefficient),
        whitened_visible_gram_matrix=np.real_if_close(whitened_visible_gram),
        singular_values=np.real_if_close(singular_values),
        restricted_singular_values=np.real_if_close(restricted_singular_values),
        whitened_singular_values=np.real_if_close(whitened_singular_values),
        visible_gram_eigenvalues=np.real_if_close(visible_gram_eigenvalues),
        whitened_gram_eigenvalues=np.real_if_close(whitened_gram_eigenvalues),
        vvr=vvr,
        ovd=ovd,
        whitened_ovd=whitened_ovd,
        soft_ovd=soft_ovd,
        latent_dim=int(latent_basis_matrix.shape[1]),
        visible_effective_rank=effective_rank_psd(visible_gram, tol=algebraic_tol),
        whitened_effective_rank=effective_rank_psd(whitened_visible_gram, tol=algebraic_tol),
        principal_angles_rad=np.real_if_close(angles),
        principal_angles_deg=np.real_if_close(np.degrees(angles)),
        noise_threshold=float(noise_threshold),
    )


class IsingVolterraAnalyzer:
    """High-level orchestrator for the dense PTM/Volterra diagnostics."""

    def __init__(
        self,
        model: ReservoirModelProtocol,
        observables: Sequence[np.ndarray] | None = None,
        max_order: int = 2,
        lag_horizon: int = 2,
        fd_step: float = 5e-3,
        algebraic_tol: float = 1e-9,
        expansion_point: float = 0.0,
    ):
        self.model = model
        self.max_order = max_order
        self.lag_horizon = lag_horizon
        self.algebraic_tol = algebraic_tol
        self.expansion_point = float(expansion_point)
        self.observables = list(observables) if observables is not None else model.default_memory_observables()
        self.expansion = PTMAffineExpansion(
            model,
            max_order=max_order,
            fd_step=fd_step,
            expansion_point=expansion_point,
        )
        self.generator = TruncatedVolterraGenerator(
            self.expansion,
            max_order=max_order,
            lag_horizon=lag_horizon,
        )

    def analyze(
        self,
        n_shots: int = 2000,
        delta: float = 0.05,
        noise_scale: float = 1.0,
        covariance_model: float | np.ndarray | None = None,
    ) -> VolterraResult:
        monomials, K = self.generator.generate(tol=self.algebraic_tol)
        ensure_finite("latent kernel matrix", K)
        # Example dense output:
        # ``monomials = [(1, 0), (0, 1), (2, 0), (1, 1)]``, ``K.shape = (15, 4)``.

        tau = _noise_threshold(n_shots=n_shots, delta=delta, noise_scale=noise_scale)
        covariance = _coerce_covariance_matrix(covariance_model, len(self.observables), tau)

        if K.shape[1] == 0:
            empty_latent = np.zeros((self.model.dim_memory**2 - 1, 0), dtype=complex)
            return _empty_result(monomials, empty_latent, covariance, tau)

        # ``K`` is only a spanning family; QR produces the orthonormal basis needed
        # for the restricted matrix ``O_{P,L}`` and for principal-angle geometry.
        latent_basis = orthonormal_basis_from_columns(K, tol=self.algebraic_tol)
        readout = self.model.readout_matrix(self.observables)
        ensure_finite("readout matrix", readout)
        # Example shapes for ``n_memory = 2``, two observables:
        # ``latent_basis.shape == (15, r)``, ``readout.shape == (2, 15)``.

        # ``readout_matrix`` stores tr(M_j P_mu) / d_M, while the paper's visible
        # matrices use the raw readout map X -> tr(M_j X). Multiply back by d_M.
        visible_coefficient = ensure_finite(
            "visible coefficient matrix",
            self.model.dim_memory * checked_matmul("readout @ latent kernel matrix", readout, K),
        )
        restricted = ensure_finite(
            "restricted measurement matrix",
            self.model.dim_memory * checked_matmul("readout @ latent basis matrix", readout, latent_basis),
        )
        readout_nullspace = null_space(readout, tol=self.algebraic_tol)
        # Example:
        # ``visible_coefficient.shape == (2, 4)``, ``restricted.shape == (2, r)``.

        return _finalize_result(
            monomials=monomials,
            latent_kernel_matrix=K,
            latent_basis_matrix=latent_basis,
            restricted_measurement_matrix=restricted,
            visible_coefficient_matrix=visible_coefficient,
            readout_nullspace_matrix=readout_nullspace,
            covariance_matrix=covariance,
            noise_threshold=tau,
            algebraic_tol=self.algebraic_tol,
        )


class ReducedVolterraBasisBuilder:
    """Generate the exact ``(P, L)``-truncated reduced observable sector.

    The generated span is the paper's observable-side sector

    ``span((Phi_0^*)^m_p M_qp ... M_q1 (Phi_0^*)^m_0(O))``

    with total insertion order at most ``P`` and total drift count at most ``L``.
    The internal word labels encode one zero-input drift by ``0`` and an input
    insertion of order ``q`` by the integer ``q``. The leading entry identifies
    the seed observable index (1-based).
    """

    def __init__(
        self,
        model: ReservoirModelProtocol,
        seed_observables: Sequence[np.ndarray],
        max_order: int,
        lag_horizon: int,
        fd_step: float = 5e-3,
        algebraic_tol: float = 1e-9,
        stencil_radius: int | None = None,
        max_basis_size: int | None = None,
        expansion_point: float = 0.0,
    ):
        self.model = model
        self.seed_observables = list(seed_observables)
        self.max_order = max_order
        self.lag_horizon = lag_horizon
        self.fd_step = fd_step
        self.algebraic_tol = algebraic_tol
        self.stencil_radius = stencil_radius if stencil_radius is not None else max(2, max_order + 1)
        self.max_basis_size = max_basis_size
        self.expansion_point = float(expansion_point)

    def build(self) -> Tuple[List[FeatureLabel], List[np.ndarray]]:
        basis: List[np.ndarray] = []
        labels: List[FeatureLabel] = []
        stack: List[_ReducedWordState] = []

        for seed_index, observable in enumerate(self.seed_observables):
            # Example seed states for ``observables = [Z0, Z1]``:
            # first push ``(seed_index=0, word=())``, then ``(seed_index=1, word=())``.
            stack.append(
                _ReducedWordState(
                    operator=np.asarray(observable, dtype=complex),
                    total_order=0,
                    total_drift=0,
                    seed_index=seed_index,
                    word=(),
                )
            )

        while stack:
            state = stack.pop()
            # Example popped state:
            # ``seed_index = 1``, ``total_order = 1``, ``total_drift = 1``,
            # ``word = (0, 1)`` means "start from observable 2, drift once, insert once".
            if state.total_order > 0:
                # Only retain words with at least one insertion. Pure drift words
                # correspond to the order-zero background/readout sector, whereas
                # the truncated Volterra family of interest starts at total order 1.
                orth = orthogonalize_operator(state.operator, basis, tol=self.algebraic_tol)
                if orth is not None:
                    basis.append(orth)
                    labels.append((state.seed_index + 1, *state.word))
                    # Example stored label: ``(2, 0, 1)``.
                    if self.max_basis_size is not None and len(basis) >= self.max_basis_size:
                        break

            if state.total_drift < self.lag_horizon:
                # Appending ``0`` to the word means "apply one more zero-input
                # adjoint step". Enumerating these steps explicitly makes the code
                # follow the paper's ``sum m_j <= L`` truncation literally.
                stack.append(
                    _ReducedWordState(
                        operator=self.model.channel_adjoint(self.expansion_point, state.operator),
                        total_order=state.total_order,
                        total_drift=state.total_drift + 1,
                        seed_index=state.seed_index,
                        word=state.word + (0,),
                    )
                )
                # Example: ``word = (1,)`` becomes ``(1, 0)`` after one extra drift.

            remaining_order = self.max_order - state.total_order
            for q in range(remaining_order, 0, -1):
                # Appending ``q`` means "apply the q-th insertion superoperator".
                # The finite-difference derivative implements the paper's ``M_q``
                # without ever building a dense PTM/Liouville matrix.
                stack.append(
                    _ReducedWordState(
                        operator=self.model.channel_derivative_adjoint(
                            q,
                            state.operator,
                            u0=self.expansion_point,
                            fd_step=self.fd_step,
                            stencil_radius=self.stencil_radius,
                        ),
                        total_order=state.total_order + q,
                        total_drift=state.total_drift,
                        seed_index=state.seed_index,
                        word=state.word + (q,),
                    )
                )
                # Example with ``remaining_order = 2``:
                # from ``word = (0,)`` we enqueue ``(0, 2)`` and ``(0, 1)``.

        return labels, basis


class ReducedVolterraAnalyzer:
    """Observable-side reduced Volterra diagnostics without dense PTM construction."""

    def __init__(
        self,
        model: ReservoirModelProtocol,
        observables: Sequence[np.ndarray] | None = None,
        max_order: int = 2,
        lag_horizon: int = 2,
        fd_step: float = 5e-3,
        algebraic_tol: float = 1e-9,
        max_basis_size: int | None = None,
        expansion_point: float = 0.0,
    ):
        self.model = model
        self.max_order = max_order
        self.lag_horizon = lag_horizon
        self.fd_step = fd_step
        self.algebraic_tol = algebraic_tol
        self.max_basis_size = max_basis_size
        self.expansion_point = float(expansion_point)
        self.observables = list(observables) if observables is not None else model.default_memory_observables()
        self.builder = ReducedVolterraBasisBuilder(
            model=model,
            seed_observables=self.observables,
            max_order=max_order,
            lag_horizon=lag_horizon,
            fd_step=fd_step,
            algebraic_tol=algebraic_tol,
            max_basis_size=max_basis_size,
            expansion_point=expansion_point,
        )

    def _restricted_measurement_matrix(self, basis_ops: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(self.observables), len(basis_ops)), dtype=complex)
        for j, observable in enumerate(self.observables):
            for ell, basis_op in enumerate(basis_ops):
                # Each entry is the raw readout functional ``tr(M_j H_ell)`` in
                # Hilbert-Schmidt form, matching the paper's reduced matrix.
                out[j, ell] = hs_inner_product(observable, basis_op)
                # Example: ``out[0, 3]`` is the overlap between observable ``Z0``
                # and reduced basis operator ``H_3``.
        return ensure_finite("restricted measurement matrix", out)

    def analyze(
        self,
        n_shots: int = 2000,
        delta: float = 0.05,
        noise_scale: float = 1.0,
        covariance_model: float | np.ndarray | None = None,
    ) -> VolterraResult:
        monomials, basis_ops = self.builder.build()
        # Example reduced output:
        # ``monomials = [(2, 1), (2, 0, 1), (1, 2)]`` and ``len(basis_ops) == 3``.
        tau = _noise_threshold(n_shots=n_shots, delta=delta, noise_scale=noise_scale)
        covariance = _coerce_covariance_matrix(covariance_model, len(self.observables), tau)

        if not basis_ops:
            empty_latent = np.zeros((self.model.dim_memory**2, 0), dtype=complex)
            return _empty_result(monomials, empty_latent, covariance, tau)

        restricted = self._restricted_measurement_matrix(basis_ops)
        # The reduced builder already returns an orthonormal operator basis, so
        # flattening it gives both the ambient-space basis for angle diagnostics
        # and the latent-basis matrix consumed by the shared evaluation layer.
        latent_columns = np.column_stack([np.asarray(op, dtype=complex).reshape(-1) for op in basis_ops])
        # Example shapes with ``n_memory = 2`` and 3 reduced basis operators:
        # ``restricted.shape == (2, 3)``, ``latent_columns.shape == (16, 3)``.
        readout_nullspace = null_space(_ambient_readout_matrix(self.observables), tol=self.algebraic_tol)

        # In the reduced methodology the orthonormalized basis itself is the
        # restricted-side basis, so the kernel-side and restricted-side visible
        # matrices coincide.
        return _finalize_result(
            monomials=monomials,
            latent_kernel_matrix=latent_columns,
            latent_basis_matrix=latent_columns,
            restricted_measurement_matrix=restricted,
            visible_coefficient_matrix=restricted,
            readout_nullspace_matrix=readout_nullspace,
            covariance_matrix=covariance,
            noise_threshold=tau,
            algebraic_tol=self.algebraic_tol,
        )


__all__ = [
    "IsingVolterraAnalyzer",
    "PTMAffineExpansion",
    "ReducedVolterraAnalyzer",
    "ReducedVolterraBasisBuilder",
    "TruncatedVolterraGenerator",
    "VolterraResult",
]
