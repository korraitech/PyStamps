"""CPU controls for the Step 2 shared-baseline trial-search experiment.

Only finite, nonzero phase samples with a common nonconstant baseline are in
scope. The production routine masks zero samples per pixel, changing its trial
matrix; silently using shared weights there would benchmark another algorithm.
Matrix multiplication changes summation order. FP32 and near/exact ties can
therefore select a different trial, even when sums are numerically close.
"""

from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np


def fixture(n_pixels, n_ifg=60, n_trial_wraps=2.0, seed=2005):
    """Return (complex128 phase, float64 baseline, wraps), using Step 2 layout."""
    if n_pixels < 0 or n_ifg < 2:
        raise ValueError("need nonnegative pixels and at least two interferograms")
    rng = np.random.RandomState(seed)
    phase = np.exp(1j * (2.0 * np.pi * rng.rand(n_ifg, n_pixels)).T)
    bperp = np.random.RandomState(seed + 1).uniform(-300, 300, n_ifg)
    _baseline(bperp, n_trial_wraps)
    return phase, bperp, n_trial_wraps


def _baseline(bperp, wraps):
    bperp = np.asarray(bperp)
    if (bperp.ndim != 1 or len(bperp) < 2 or not np.isrealobj(bperp)
            or not np.all(np.isfinite(bperp)) or np.ptp(bperp) <= 0):
        raise ValueError("baseline must be finite, real, nonconstant and 1D")
    if not np.isfinite(wraps) or wraps < 0:
        raise ValueError("wraps must be finite and nonnegative")
    return bperp


def _phase(phase, n_ifg):
    phase = np.asarray(phase)
    if phase.ndim != 2 or phase.shape[1] != n_ifg:
        raise ValueError("phase must have shape (pixels, interferograms)")
    if not np.all(np.isfinite(phase)) or np.any(phase == 0):
        raise ValueError("phase must be finite and nonzero; per-pixel masks are out of scope")
    return phase


def trial_weights(bperp, wraps):
    """Build the source's (interferograms, trials) complex128 phasor matrix."""
    bperp = _baseline(bperp, wraps).astype(np.float64, copy=False)
    trial_mult = np.arange(-np.ceil(8 * wraps), np.ceil(8 * wraps) + 1)
    trial_phase = bperp / (np.max(bperp) - np.min(bperp)) * np.pi / 4
    return np.exp(-1j * np.outer(trial_phase, trial_mult))


def source_search(phase, bperp, wraps):
    """Repeat the original per-pixel trial setup, sum, coherence, and argmax."""
    bperp = _baseline(bperp, wraps)
    phase = _phase(phase, len(bperp))
    n_trials = int(2 * np.ceil(8 * wraps) + 1)
    sums = np.empty((len(phase), n_trials), dtype=np.complex128)
    best = np.empty(len(phase), dtype=np.intp)
    for row, cpxphase in enumerate(phase):
        # Keep setup inside this loop: it is repeated in the actual source.
        bperp_range = np.max(bperp) - np.min(bperp)
        trial_mult = np.arange(-np.ceil(8 * wraps), np.ceil(8 * wraps) + 1)
        trial_phase = bperp / bperp_range * np.pi / 4
        trial_phase_mat = np.exp(-1j * np.outer(trial_phase, trial_mult))
        cpxphase_mat = np.tile(cpxphase.reshape(-1, 1), (1, n_trials))
        phaser_sum = np.sum(trial_phase_mat * cpxphase_mat, axis=0)
        coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))
        sums[row] = phaser_sum
        best[row] = np.argmax(coh_trial)
    return sums, best


def batched_search(phase, weights, chunk_size=4096):
    """Precomputed matmul search; complex64 inputs keep complex64 arithmetic.

    Scratch space is bounded by a chunk of (pixels, trials), never (N,I,T).
    The full (N,T) sums are returned for validation. NumPy argmax selects the
    first exact maximum; floating-point changes can still break near ties.
    """
    weights = np.asarray(weights)
    if weights.ndim != 2 or not weights.shape[1] or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be a finite (interferograms, trials) matrix")
    phase = _phase(phase, weights.shape[0])
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    sums = np.empty((len(phase), weights.shape[1]), dtype=np.result_type(phase, weights))
    best = np.empty(len(phase), dtype=np.intp)
    for start in range(0, len(phase), chunk_size):
        stop = min(start + chunk_size, len(phase))
        sums[start:stop] = phase[start:stop] @ weights
        coherence = np.abs(sums[start:stop]) / np.sum(np.abs(phase[start:stop]), axis=1)[:, None]
        best[start:stop] = np.argmax(coherence, axis=1)
    return sums, best


@lru_cache(maxsize=1)
def _reference_function():
    path = Path(__file__).resolve().parents[2] / "src" / "step" / "ps_topofit.py"
    spec = importlib.util.spec_from_file_location("_experiment_ps_topofit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ps_topofit


def reference_topofit(phase, bperp, wraps):
    """Call actual ps_topofit without importing SciPy or the application."""
    return _reference_function()(phase, bperp, wraps)


def refine(phase, bperp, wraps, best):
    """Run the original lstsq refinement from arbitrary selected trial indices."""
    bperp = _baseline(bperp, wraps)
    phase = _phase(phase, len(bperp))
    trial_mult = np.arange(-np.ceil(8 * wraps), np.ceil(8 * wraps) + 1)
    best = np.asarray(best)
    if (best.shape != (len(phase),) or not np.issubdtype(best.dtype, np.integer)
            or np.any(best < 0) or np.any(best >= len(trial_mult))):
        raise ValueError("best must contain one valid integer trial index per pixel")
    result = {key: np.empty(len(phase), dtype=np.float64) for key in ("K", "C", "coh")}
    result["phase_residual"] = np.empty(phase.shape, dtype=np.complex128)
    bperp_range = np.max(bperp) - np.min(bperp)
    for row, cpxphase in enumerate(phase):
        K0 = np.pi / 4 / bperp_range * trial_mult[best[row]]
        resphase = cpxphase * np.exp(-1j * (K0 * bperp))
        offset_phase = np.sum(resphase)
        resphase = np.angle(resphase * np.conj(offset_phase))
        weighting = np.abs(cpxphase)
        mopt = np.linalg.lstsq(
            np.diag(weighting) @ bperp.reshape(-1, 1),
            np.diag(weighting) @ resphase.reshape(-1, 1),
            rcond=None,
        )[0][0]
        K0 = K0 + mopt
        phase_residual = cpxphase * np.exp(-1j * (K0 * bperp))
        mean_phase_residual = np.sum(phase_residual)
        result["K"][row] = K0[0]
        result["C"][row] = np.angle(mean_phase_residual)
        result["coh"][row] = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))
        result["phase_residual"][row] = phase_residual
    return result
