import torch
from typing import Tuple

# This is the torch version of ps_topofit. It is faster than the numpy version, but does not have a plot function.
def ps_topofit_torch(cpxphase: torch.Tensor, 
                          bperp: torch.Tensor, 
                          n_trial_wraps: float, 
                          plotflag: str = 'n',
                          asym: float = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched version of ps_topofit using PyTorch for GPU acceleration.
    
    Args:
        cpxphase: Complex phase values as a PyTorch tensor [batch_size, n_ifg]
        bperp: Perpendicular baselines as a PyTorch tensor [batch_size, n_ifg]
        n_trial_wraps: Number of trial wraps
        plotflag: Plot flag ('y' or 'n')
        asym: Asymmetry (-1 for only -ve K, +1 for only +ve K, default 0)
    
    Returns:
        Tuple containing:
        - K0: Best-fitting K values [batch_size]
        - C0: Phase offsets [batch_size]
        - coh0: Coherence values [batch_size]
        - phase_residual: Residual phases [batch_size, n_ifg]
    """
    batch_size = cpxphase.shape[0]
    device = cpxphase.device

    # Filter non-zero values using absolute value for complex numbers
    ix = torch.abs(cpxphase) != 0
    n_ix = ix.sum(dim=1)  # [batch_size]

    # Handle zero cases
    zero_mask = (n_ix == 0)
    if zero_mask.any():
        K0 = torch.zeros(batch_size, device=device, dtype=torch.float64)
        C0 = torch.zeros(batch_size, device=device, dtype=torch.float64)
        coh0 = torch.zeros(batch_size, device=device, dtype=torch.float64)
        phase_residual = torch.zeros_like(cpxphase)
        if zero_mask.all():
            return K0, C0, coh0, phase_residual

    # Get bperp range for each sample in batch
    bperp_range = torch.max(bperp, dim=1)[0] - torch.min(bperp, dim=1)[0]  # [batch_size]
    zero_range_mask = (bperp_range == 0)

    # Set K search range based on asymmetry parameter
    n_trial_wraps_tensor = torch.tensor(n_trial_wraps, dtype=torch.float64, device=device)
    trial_mult = torch.arange(
        -int(torch.ceil(8 * n_trial_wraps_tensor).item()),
        int(torch.ceil(8 * n_trial_wraps_tensor).item()) + 1,
        dtype=torch.int,
        device=device
    ) + int(asym * 8 * n_trial_wraps)

    if asym > 0:
        trial_mult = trial_mult[trial_mult >= 0]
    elif asym < 0:
        trial_mult = trial_mult[trial_mult <= 0]

    n_trials = len(trial_mult)

    # Calculate coherence for each trial value for all samples simultaneously
    # [batch_size, 1] / [batch_size, 1] * pi/4 -> [batch_size, 1]
    trial_phase = (bperp / bperp_range.unsqueeze(1)) * (torch.pi / 4)
    
    # [batch_size, n_ifg, n_trials]
    trial_phase_mat = torch.exp(-1j * torch.einsum('bi,t->bit', trial_phase, trial_mult))
    
    # [batch_size, n_ifg, n_trials]
    cpxphase_mat = cpxphase.unsqueeze(-1).expand(-1, -1, n_trials)
    
    # Calculate coherence for all trials and all samples
    phaser = trial_phase_mat * cpxphase_mat
    phaser_sum = torch.sum(phaser, dim=1)  # [batch_size, n_trials]
    coh_trial = torch.abs(phaser_sum) / torch.sum(torch.abs(cpxphase), dim=1, keepdim=True)

    # Find maximum coherence for each sample
    i_max = torch.argmax(coh_trial, dim=1)  # [batch_size]
    K0 = torch.pi/4 * trial_mult[i_max] / bperp_range

    # Calculate initial phase offset
    exp_term = torch.exp(-1j * (K0.unsqueeze(1) * bperp))
    C0 = torch.angle(torch.sum(cpxphase * exp_term, dim=1))
    coh0 = torch.max(coh_trial, dim=1)[0]

    # Refine K estimate
    resphase = cpxphase * exp_term
    offset_phase = torch.sum(resphase, dim=1, keepdim=True)
    resphase = torch.angle(resphase * torch.conj(offset_phase))
    weighting = torch.abs(cpxphase)

    # Solve weighted least squares for each sample
    weighted_bperp = weighting * bperp
    weighted_resphase = weighting * resphase
    
    # Handle each sample separately for least squares
    mopt = torch.zeros(batch_size, device=device, dtype=torch.float64)
    for i in range(batch_size):
        if not zero_mask[i] and not zero_range_mask[i]:
            solution = torch.linalg.lstsq(
                weighted_bperp[i:i+1].T,
                weighted_resphase[i:i+1].T
            ).solution
            mopt[i] = solution[0].item()

    # Update K0 and calculate final residuals
    K0 = K0 + mopt
    phase_residual = cpxphase * torch.exp(-1j * (K0.unsqueeze(1) * bperp))
    mean_phase_residual = torch.sum(phase_residual, dim=1)
    C0 = torch.angle(mean_phase_residual)
    coh0 = torch.abs(mean_phase_residual) / torch.sum(torch.abs(phase_residual), dim=1)

    # Handle special cases
    if zero_range_mask.any():
        coh0[zero_range_mask] = 1.0
        phase_residual[zero_range_mask] = cpxphase[zero_range_mask]

    if plotflag == 'y':
        # Note: plotting not implemented for batch processing
        print("Warning: Plotting not implemented for batch processing")

    return K0, C0, coh0, phase_residual