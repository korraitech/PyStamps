import torch
from typing import Tuple

# This is the torch version of ps_topofit. It is faster than the numpy version, but does not have a plot function.
def ps_topofit_torch2(cpxphase: torch.Tensor, 
                bperp: torch.Tensor, 
                n_trial_wraps: float,
                asym: float = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched version of ps_topofit using PyTorch for GPU acceleration. Optimized for performance.
    
    Args:
        cpxphase: Complex phase values as a PyTorch tensor [batch_size, n_ifg]
        bperp: Perpendicular baselines as a PyTorch tensor [batch_size, n_ifg]
        n_trial_wraps: Number of trial wraps
        asym: Asymmetry (-1 for only -ve K, +1 for only +ve K, default 0)
    
    Returns:
        Tuple containing:
        - K0: Best-fitting K values [batch_size]
        - C0: Phase offsets [batch_size]
        - coh0: Coherence values [batch_size]
        - phase_residual: Residual phases [batch_size, n_ifg]
    """
    # print(cpxphase.shape)
    # print(bperp.shape)
    # print(n_trial_wraps)
    # print(asym)
    batch_size = cpxphase.shape[0]
    device = cpxphase.device

    # Filter out zero-valued rows
    ix = torch.abs(cpxphase) != 0
    n_ix = ix.sum(dim=1)  # [batch_size]
    zero_mask = (n_ix == 0)

    # Pre-allocate zero outputs for completely-zero rows
    K0 = torch.zeros(batch_size, device=device, dtype=torch.float64)
    C0 = torch.zeros(batch_size, device=device, dtype=torch.float64)
    coh0 = torch.zeros(batch_size, device=device, dtype=torch.float64)
    phase_residual = torch.zeros_like(cpxphase)

    # Early return if all rows are zero
    if zero_mask.all():
        return K0, C0, coh0, phase_residual

    # Get bperp range for each sample, detect zero-range rows
    bperp_range = torch.max(bperp, dim=1)[0] - torch.min(bperp, dim=1)[0]  # [batch_size]
    zero_range_mask = (bperp_range == 0)

    # Set up trial multiples (integer increments * 8 for step in K)
    n_trial_wraps_tensor = torch.tensor(n_trial_wraps, dtype=torch.float64, device=device)
    trial_mult = torch.arange(
        -int(torch.ceil(8 * n_trial_wraps_tensor).item()),
         int(torch.ceil(8 * n_trial_wraps_tensor).item()) + 1,
        dtype=torch.int,
        device=device
    ) + int(asym * 8 * n_trial_wraps)

    # Apply asym > 0 or asym < 0 constraints
    if asym > 0:
        trial_mult = trial_mult[trial_mult >= 0]
    elif asym < 0:
        trial_mult = trial_mult[trial_mult <= 0]

    n_trials = len(trial_mult)

    # Avoid division by zero for the rows with bperp_range=0
    safe_bperp_range = torch.where(zero_range_mask, torch.ones_like(bperp_range), bperp_range)

    # Trial phase: normalized by bperp_range, scaled by pi/4
    # shape: [batch_size, n_ifg]
    trial_phase = (bperp / safe_bperp_range.unsqueeze(1)) * (torch.pi / 4)

    # Expand to [batch_size, n_ifg, n_trials], compute e^{-i * trial_phase * trial_mult}
    # This uses Einstein summation to broadcast 'trial_mult' over the second dimension
    trial_phase_mat = torch.exp(-1j * torch.einsum('bi,t->bit', trial_phase, trial_mult))

    # Broadcast cpxphase to [batch_size, n_ifg, n_trials]
    cpxphase_mat = cpxphase.unsqueeze(-1).expand(-1, -1, n_trials)

    # Multiply and sum across ifgs => sum of phaser
    phaser = trial_phase_mat * cpxphase_mat  # [batch_size, n_ifg, n_trials]
    phaser_sum = torch.sum(phaser, dim=1)    # [batch_size, n_trials]

    # Compute coherence for each trial
    cpxphase_abs_sum = torch.sum(torch.abs(cpxphase), dim=1, keepdim=True)  # [batch_size, 1]
    coh_trial = torch.abs(phaser_sum) / torch.where(cpxphase_abs_sum == 0,
                                                    torch.ones_like(cpxphase_abs_sum), 
                                                    cpxphase_abs_sum)

    # Argmax coherence
    i_max = torch.argmax(coh_trial, dim=1)  # [batch_size]
    # K0 init from best trial, skipping zero-range rows
    K0 = (torch.pi / 4) * trial_mult[i_max] / safe_bperp_range

    # Exponent to remove the best linear phase from cpxphase => for computing offset
    exp_term = torch.exp(-1j * (K0.unsqueeze(1) * bperp))
    cpxphase_exp = cpxphase * exp_term
    # Phase offset
    C0 = torch.angle(torch.sum(cpxphase_exp, dim=1))
    # Coherence from best trial
    coh0 = torch.max(coh_trial, dim=1)[0]

    # Residual phase after removing common offset
    offset_phase = torch.sum(cpxphase_exp, dim=1, keepdim=True)
    resphase = torch.angle(cpxphase_exp * torch.conj(offset_phase))

    # Weighted least squares to refine K in vectorized form
    # Weighted by |cpxphase|.
    weighting = torch.abs(cpxphase)
    weighted_bperp = weighting * bperp

    # Weighted sums
    w_num = torch.sum(weighted_bperp * resphase, dim=1)         # ∑(w_i * x_i * y_i)
    w_den = torch.sum(weighted_bperp * bperp, dim=1)            # ∑(w_i * x_i^2)

    # Avoid division by zero for rows with zero bperp range or all zero
    non_zero_mask = ~zero_mask & ~zero_range_mask
    mopt = torch.zeros_like(K0)
    mopt[non_zero_mask] = w_num[non_zero_mask] / w_den[non_zero_mask]

    # Update K0
    K0 = K0 + mopt

    # Final residual and coherence
    phase_residual = cpxphase * torch.exp(-1j * (K0.unsqueeze(1) * bperp))
    mean_phase_residual = torch.sum(phase_residual, dim=1)
    C0 = torch.angle(mean_phase_residual)
    coh0 = torch.abs(mean_phase_residual) / torch.sum(torch.abs(phase_residual), dim=1)

    # Handle zero-range rows
    if zero_range_mask.any():
        coh0[zero_range_mask] = 1.0
        phase_residual[zero_range_mask] = cpxphase[zero_range_mask]

    return K0, C0, coh0, phase_residual
