import torch
import numpy as np
from .ps_topofit_torch2 import ps_topofit_torch2

def random_coh(rand_ifg: np.array, bperp: np.array, n_trial_wraps: int, batch_size: int = 1000) -> np.array:
    """
    Compute coherence using PyTorch with true batch processing.
    
    Args:
        rand_ifg (np.array): Array of random interferogram phases. Shape: [n_rand, n_ifg]
        bperp (np.array): Array of perpendicular baseline values. Shape: [n_ifg]
        n_trial_wraps (int): Number of trial wraps for ps_topofit.
        batch_size (int): Optional batch size for processing on the GPU. Default=1000.

    Returns:
        np.array: An array of coherence values for each row in rand_ifg.
    """
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input arrays to float32 for speed & memory efficiency
    # (If double precision is required, switch dtype to torch.float64)
    rand_ifg_tensor = torch.tensor(rand_ifg, dtype=torch.float32, device=device)
    bperp_tensor = torch.tensor(bperp, dtype=torch.float32, device=device)

    # Convert all random phases to complex exponentials at once
    # This uses PyTorch's complex type
    exp_rand_ifg = torch.exp(1j * rand_ifg_tensor.to(torch.complex64))

    n_rand = rand_ifg_tensor.shape[0]

    # Prepare tensor to store coherence results
    coh_rand_tensor = torch.empty(n_rand, dtype=torch.float32, device=device)

    # Loop over batches (helpful if n_rand is large)
    for batch_start in range(0, n_rand, batch_size):
        batch_end = min(batch_start + batch_size, n_rand)
        
        # Slice the precomputed complex exponentials
        batch_exp_rand_ifg = exp_rand_ifg[batch_start:batch_end]
        
        # Expand bperp for batch processing: [batch_size, n_ifg]
        batch_bperp = bperp_tensor.unsqueeze(0).expand(batch_end - batch_start, -1)

        # Compute coherence via ps_topofit
        _, _, coh_batch, _ = ps_topofit_torch2(
            batch_exp_rand_ifg,  # [batch_size, n_ifg, complex]
            batch_bperp,         # [batch_size, n_ifg]
            n_trial_wraps
        )

        # Save results for the current batch
        coh_rand_tensor[batch_start:batch_end] = coh_batch

    # Move the coherence data back to CPU and convert to numpy
    return coh_rand_tensor.cpu().numpy()
