import numpy as np

def ps_topofit(cpxphase:np.ndarray, bperp:np.ndarray, n_trial_wraps:float):
    # if cpxphase.shape[1] > 1:
    #     cpxphase = cpxphase.T

    ix = cpxphase != 0  # if signal of one image is 0, dph set to 0
    cpxphase = cpxphase[ix]
    bperp = bperp[ix]
    bperp_range = np.max(bperp) - np.min(bperp)
    
    trial_mult = np.arange(-np.ceil(8 * n_trial_wraps), np.ceil(8 * n_trial_wraps) + 1)
    n_trials = len(trial_mult)
    trial_phase = bperp / bperp_range * np.pi / 4
    trial_phase_mat = np.exp(-1j * np.outer(trial_phase, trial_mult))
    cpxphase_mat = np.tile(cpxphase.reshape(-1, 1), (1, n_trials))
    phaser = trial_phase_mat * cpxphase_mat
    phaser_sum = np.sum(phaser, axis=0)
    C_trial = np.angle(phaser_sum)
    coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))
    
    coh_high_max_ix = np.argmax(coh_trial)

    K0 = np.pi / 4 / bperp_range * trial_mult[coh_high_max_ix]
    C0 = C_trial[coh_high_max_ix]
    coh0 = coh_trial[coh_high_max_ix]

    resphase = cpxphase * np.exp(-1j * (K0 * bperp))
    offset_phase = np.sum(resphase)
    resphase = np.angle(resphase * np.conj(offset_phase))
    weighting = np.abs(cpxphase)
    mopt = np.linalg.lstsq(
        np.diag(weighting) @ bperp.reshape(-1, 1), 
        np.diag(weighting) @ resphase.reshape(-1, 1), 
        rcond=None
    )[0][0]
    K0 = K0 + mopt
    phase_residual = cpxphase * np.exp(-1j * (K0 * bperp))
    mean_phase_residual = np.sum(phase_residual)
    C0 = np.angle(mean_phase_residual)
    coh0 = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))
    return K0, C0, coh0, phase_residual