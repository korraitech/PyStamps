import numpy as np
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt

def ps_topofit(cpxphase: np.ndarray, 
               bperp: np.ndarray, 
               n_trial_wraps: float, 
               plotflag: str = 'n',
               asym: float = 0) -> Tuple[float, float, float, np.ndarray]:
    """
    Find best-fitting range error.
    
    Args:
        cpxphase: Complex phase values
        bperp: Perpendicular baselines
        n_trial_wraps: Number of trial wraps
        plotflag: Plot flag ('y' or 'n')
        asym: Asymmetry (-1 for only -ve K, +1 for only +ve K, default 0)
    
    Returns:
        Tuple containing:
        - K0: Best-fitting K value
        - C0: Phase offset
        - coh0: Coherence value
        - phase_residual: Residual phase
    """
    # Add this where you want to pause execution
    # print("zuzu cpxphase: ", cpxphase, cpxphase.shape,cpxphase.dtype.fields)
    # Convert structured array to complex array if needed
    if cpxphase.dtype.fields is not None:
        real_part = cpxphase['real']
        imag_part = cpxphase['imag']
        cpxphase = real_part + 1j * imag_part


    # Ensure cpxphase is a column vector
    if cpxphase.ndim > 1 and cpxphase.shape[1] > 1:
        cpxphase = cpxphase.T

    # Filter non-zero values using absolute value for complex numbers
    ix = np.abs(cpxphase) != 0  # if signal of one image is 0, dph set to 0
    cpxphase = cpxphase[ix]
    bperp = bperp[ix]
    n_ix = len(ix)

    if n_ix == 0:
        return 0, 0, 0, np.zeros_like(cpxphase)

    # Get wrapped phase
    wphase = np.angle(cpxphase)

    # Set range for K search
    bperp_range = np.max(bperp) - np.min(bperp)
    if bperp_range == 0:
        return 0, 0, 1, cpxphase

    # Set K search range based on asymmetry parameter
    trial_mult = np.arange(-int(np.ceil(8 * n_trial_wraps)), int(np.ceil(8 * n_trial_wraps)) + 1, dtype=int) + int(asym * 8 * n_trial_wraps)
    if asym > 0:
        trial_mult = trial_mult[trial_mult >= 0]
    elif asym < 0:
        trial_mult = trial_mult[trial_mult <= 0]

    # Initialize arrays for coherence calculation
    n_trials = len(trial_mult)
    coh_trial = np.zeros(n_trials)

    # Ensure bperp is a row vector
    if bperp.ndim > 1 and bperp.shape[1] == 1:
        bperp = bperp.flatten()

    # Calculate coherence for each trial value - Current version
    for i in range(n_trials):
        pexp = np.exp(-1j * trial_mult[i] * np.pi/4/bperp_range * bperp)
        coh_trial[i] = np.abs(np.sum(cpxphase * np.conj(pexp))) / n_ix

    # Should be replaced with:
    trial_phase = bperp/bperp_range * np.pi/4
    trial_phase_mat = np.exp(-1j * np.outer(trial_phase, trial_mult))
    # crr till here zuzu
    cpxphase_mat = np.tile(cpxphase, (n_trials, 1)).T
    phaser = trial_phase_mat * cpxphase_mat
    phaser_sum = np.sum(phaser, axis=0)
    coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))

    # Find maximum coherence and calculate initial values
    i_max = np.argmax(coh_trial)
    K0 = np.pi/4/bperp_range * trial_mult[i_max]  # Initial K0 calculation
    C0 = np.angle(np.sum(cpxphase * np.exp(-1j * K0 * bperp)))  # Matches MATLAB's C_trial(coh_high_max_ix)
    coh0 = coh_trial[i_max]

    # Refine K estimate
    resphase = cpxphase * np.exp(-1j * (K0 * bperp))  # subtract approximate fit
    offset_phase = np.sum(resphase)
    resphase = np.angle(resphase * np.conj(offset_phase))  # subtract offset, take angle (unweighted)
    weighting = np.abs(cpxphase)
    
    # Solve weighted least squares
    mopt = np.linalg.lstsq(
        (weighting * bperp).reshape(-1, 1), 
        weighting * resphase, 
        rcond=None
    )[0][0]
    
    # Update K0 and calculate final residuals in same order as MATLAB
    K0 = K0 + mopt
    phase_residual = cpxphase * np.exp(-1j * (K0 * bperp))
    mean_phase_residual = np.sum(phase_residual)
    C0 = np.angle(mean_phase_residual)
    coh0 = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))

    if plotflag == 'y':
        plot_topofit_results(bperp, wphase, K0, C0, coh_trial, trial_mult, bperp_range)

    return K0, C0, coh0, phase_residual

def plot_topofit_results(bperp, wphase, K0, C0, coh_trial, trial_mult, bperp_range):
    """Separate function for plotting topofit results."""
    
    plt.figure(figsize=(10, 8))
    
    # Coherence plot
    plt.subplot(2, 1, 1)
    plt.plot(np.pi/4/bperp_range * trial_mult, coh_trial, 'g')
    plt.ylabel('\u03B3_x')
    plt.xlabel('Spatially uncorrelated look angle error (radians/m)')
    plt.grid(True)
    plt.ylim(0, 1)
    
    # Phase plot
    plt.subplot(2, 1, 2)
    bvec = np.linspace(np.min(bperp), np.max(bperp), 200)
    wphase_hat = np.angle(np.exp(1j * (K0 * bvec + C0)))
    plt.plot(bvec, wphase_hat, 'r', linewidth=2)
    plt.plot(bperp, wphase, 'bo', linewidth=2)
    plt.ylim(-np.pi, np.pi)
    plt.ylabel('Wrapped Phase')
    plt.xlabel('B_\u27C2 (m)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

