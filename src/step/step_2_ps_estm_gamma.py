import numpy as np
import torch
import time
import os
from scipy.interpolate import interp1d
from .utils import read_h5,save_h5
from .ps_topofit_numpy import ps_topofit
from .ps_topofit_torch import ps_topofit_torch
from .clap_filt import clap_filt

def step_2_ps_estm_gamma(workdir:str,patch:str,parms:dict) -> None:
    """
    Estimate coherence of PS candidates.

    This version of ps_estm_gamma more closely follows the structure
    and variable naming/order of the original MATLAB code ps_est_gamma_quick.m.
    It extends the logic after computing Nr so that it includes patch-phase
    calculation, topographic-error estimation, iterative (re-)weighting,
    and final saving steps, analogous to the MATLAB script.  

    Args:
        workdir:str - path to the working directory
        patch:str - patch currently being processed
        parms:dict - parameters from parms.json
    """
    print("Running Step-02 ...\t[{}]".format(patch))
    print('Estimating gamma for candidate pixels...')
    patch_dir = os.path.join(workdir,patch)

    # Constants
    rho = 830000  # mean range - only approximate
    n_rand = 300000  # number of simulated random-phase pixels

    # Get parameters and extract scalar values
    grid_size = int(parms['filter_grid_size'])
    filter_weighting = parms['filter_weighting']
    n_win = int(parms['clap_win'])
    low_pass_wavelength = float(parms['clap_low_pass_wavelength'])
    clap_alpha = float(parms['clap_alpha'])
    clap_beta = float(parms['clap_beta'])
    max_topo_err = float(parms['max_topo_err'])
    wavelength = float(parms['lambda'])
    gamma_change_convergence = float(parms['gamma_change_convergence'])
    gamma_max_iterations = int(parms['gamma_max_iterations'])

    # Threshold based on baseline flag for UrbanSAR
    low_coh_thresh = 31  # equivalent to coherence = 0.31

    # Frequency setup for filtering
    freq0 = 1.0 / low_pass_wavelength
    freq_i = np.arange(
        start=-(n_win) / (grid_size * n_win) / 2.0,
        stop=((n_win - 2) / (grid_size * n_win) / 2.0) + 1e-12,
        step=1.0 / (grid_size * n_win)
    )
    butter_i = 1.0 / (1.0 + (freq_i / freq0) ** (2 * 5))
    low_pass = np.outer(butter_i, butter_i)
    low_pass = np.fft.fftshift(low_pass)

    # Load psver
    psver_data = read_h5(os.path.join(patch_dir, 'psver.h5'))
    psver = int(psver_data['psver'])

    # Build filenames
    psname = f'ps{int(psver)}.h5'
    phname = f'ph{int(psver)}.h5'
    bpname = f'bp{int(psver)}.h5'
    laname = f'la{int(psver)}.h5'
    pmname = f'pm{int(psver)}.h5'
    daname = f'da{int(psver)}.h5'

    # Load PS data
    ps_data = read_h5(os.path.join(patch_dir, psname))

    # Load baseline perpendicular matrix and version
    bp_data = read_h5(os.path.join(patch_dir,bpname))

    # Load or create D_A
    D_A = read_h5(os.path.join(patch_dir,daname))['D_A']

    # Load phase data
    ph = read_h5(os.path.join(patch_dir,phname))['ph']

    # Exclude master image from ph and bperp
    master_ix = int(ps_data['master_ix'])
    ph = np.delete(ph, master_ix, axis=1)
    bperp = np.delete(ps_data['bperp'], master_ix)
    n_ifg = int(ps_data['n_ifg'])-1
    n_ps = int(ps_data['n_ps'])
    xy = ps_data['xy']

    # Ensure 'xy' has shape (n_ps, ...)
    if xy.shape[0] != n_ps:
        xy = xy.T

    # Normalize ph
    A = np.abs(ph).astype(np.float32)
    A[A == 0] = 1.0
    ph = ph / A

    # Incidence angle
    la_data = read_h5(os.path.join(patch_dir,laname))
    inc_mean = np.mean(la_data['la']) + 0.052

    # Max K and approximate number of trial wraps
    bperp_range = float(np.max(bperp) - np.min(bperp))
    max_K = max_topo_err / (wavelength * rho * np.sin(inc_mean) / (4.0 * np.pi))
    n_trial_wraps = bperp_range * max_K / (2.0 * np.pi)
    print(f'n_trial_wraps={n_trial_wraps}')

    # Prepare for restarts
    K_ps = None
    C_ps = None
    coh_ps = None
    N_opt = None
    ph_patch = None
    ph_res = None

    # Fresh start if needed
    print('Initialising random distribution...')
    np.random.seed(2005)

    # Generate random-phase distribution
    rand_ifg = 2.0 * np.pi * np.random.rand(n_rand, n_ifg)

    # 2. PyTorch implementation
    start_time_torch = time.time()
    coh_rand_torch = compute_coh_rand_with_pytorch(rand_ifg, bperp, n_trial_wraps)
    torch_time = time.time() - start_time_torch
    print(f'PyTorch implementation took {torch_time:.2f} seconds')

    # Use the PyTorch results for subsequent processing
    coh_rand = coh_rand_torch

    # Build histogram (matching MATLAB's behavior)
    coh_bins = np.arange(0.005, 0.996, 0.01)
    bin_edges = np.arange(0, 1.001, 0.01)
    Nr, _ = np.histogram(coh_rand, bins=bin_edges, density=False)

    # Find last non-zero index more efficiently
    Nr_max_nz_ix = len(Nr) - 1 - np.argmax(Nr[::-1] != 0)

    # Initialize arrays
    step_number = 1
    n_ps_int = int(ps_data['n_ps'])
    K_ps = np.zeros(n_ps_int, dtype=np.float64)
    C_ps = np.zeros(n_ps_int, dtype=np.float64)
    coh_ps = np.zeros(n_ps_int, dtype=np.float64)
    N_opt = np.zeros(n_ps_int, dtype=np.float64)

    # ph_patch and ph_res should have shape (n_ps_int, n_ifg)
    ph_res = np.zeros((n_ps_int, n_ifg), dtype=np.float32)
    ph_patch = np.zeros((n_ps_int, n_ifg), dtype=np.complex64)

    if xy.shape[1] >= 3:
        y_vals = xy[:, 2]
        x_vals = xy[:, 1]
    else:
        y_vals = xy[:, 1]
        x_vals = xy[:, 0]

    grid_ij = np.zeros((n_ps_int, 2), dtype=int)
    grid_ij[:, 0] = np.ceil((y_vals - np.min(y_vals) + 1.0e-6) / grid_size).astype(int)
    grid_ij[:, 1] = np.ceil((x_vals - np.min(x_vals) + 1.0e-6) / grid_size).astype(int)

    # Match MATLAB boundary tweak
    max_i = np.max(grid_ij[:, 0])
    max_j = np.max(grid_ij[:, 1])
    mask_i = (grid_ij[:, 0] == max_i)
    mask_j = (grid_ij[:, 1] == max_j)
    if max_i > 0:
        grid_ij[mask_i, 0] = max_i - 1
    if max_j > 0:
        grid_ij[mask_j, 1] = max_j - 1

    i_loop = 1
    weighting = 1.0 / D_A.ravel()

    # Ensure arrays are allocated if partial or no restart
    n_ps_int = int(n_ps_int)

    # Number of grid cells
    n_i = np.max(grid_ij[:, 0]) 
    n_j = np.max(grid_ij[:, 1]) 

    print(f'{n_ps_int} PS candidates to process')
    loop_end_sw = 0
    step_number = 1

    # --------------------
    # Main iteration loop
    # --------------------
    coh_ps_temp = np.zeros(n_ps_int, dtype=np.float64)
    while loop_end_sw == 0:
        print(f'iteration #{i_loop}')
        print('Calculating patch phases...')

        # Build ph_grid & ph_filt: shape (n_i, n_j, n_ifg)
        ph_grid = np.zeros((n_i, n_j, n_ifg), dtype=np.complex64)
        ph_filt = np.zeros((n_i, n_j, n_ifg), dtype=np.complex64)

        # Reshape K_ps and weighting to (n_ps, 1) if they are 1D arrays
        K_ps = K_ps.reshape((n_ps_int, 1)) if K_ps.ndim == 1 else K_ps
        weighting = weighting.reshape((n_ps_int, 1)) if weighting.ndim == 1 else weighting

        # Vectorized ph_weight calculation
        phase_factor = np.exp(-1j * bp_data['bperp_mat'] * K_ps)
        ph_weight = ph * phase_factor * weighting

        # Accumulate into ph_grid
        for i_ps in range(n_ps_int):
            gi, gj = grid_ij[i_ps] - 1
            ph_grid[gi, gj, :] += ph_weight[i_ps, :]

        # Filter each IFG slice
        for i_ifg in range(n_ifg):
            ph_filt[:, :, i_ifg] = clap_filt(
                ph_grid[:, :, i_ifg],
                clap_alpha,
                clap_beta,
                int(n_win * 0.75),
                int(n_win * 0.25),
                low_pass
            )

        # Extract patch value per PS
        for i_ps in range(n_ps_int):
            gi, gj = grid_ij[i_ps] - 1 # MATLAB to Python index
            ph_patch[i_ps, :] = ph_filt[gi, gj, :]

        # Normalize ph_patch
        nz_idx = (np.abs(ph_patch) != 0)
        ph_patch[nz_idx] = ph_patch[nz_idx] / np.abs(ph_patch[nz_idx])

        print('Estimating topo error...')
        step_number = 2

        for i_ps in range(n_ps_int):
            psdph = ph[i_ps, :] * np.conj(ph_patch[i_ps, :])
            if np.all(psdph != 0):
                Kopt, Copt, cohopt, ph_residual = ps_topofit(
                    psdph,
                    bperp[i_ps, :] if (bperp.ndim == 2 and bperp.shape[0] == n_ps_int) else bperp,
                    n_trial_wraps,
                    'n'
                )
                # First solution
                K_ps[i_ps] = Kopt
                C_ps[i_ps] = Copt
                coh_ps[i_ps] = cohopt
                N_opt[i_ps] = 1 
                ph_res[i_ps, :] = np.angle(ph_residual)
            else:
                K_ps[i_ps] = np.nan
                coh_ps[i_ps] = 0.0

        # Return to step_number=1 after this block
        step_number = 1
        gamma_change_save = 0.0

        # Check convergence
        gamma_change_rms = np.sqrt(np.mean((coh_ps - coh_ps_temp) ** 2))
        gamma_change_change = gamma_change_rms - gamma_change_save
        print(f'gamma_change_change={gamma_change_change}')
        gamma_change_save = gamma_change_rms
        coh_ps_temp = coh_ps.copy()

        # Retrieve convergence parameters
        gamma_change_convergence = float(parms['gamma_change_convergence'])
        gamma_max_iterations = int(parms['gamma_max_iterations'])

        # Convergence condition
        if (abs(gamma_change_change) < gamma_change_convergence) or (i_loop >= gamma_max_iterations):
            loop_end_sw = 1
        else:
            i_loop += 1
            # Reweighting
            if str(filter_weighting).lower().startswith('p-square'):
                Na, _ = np.histogram(coh_ps, bins=bin_edges, density=False)
                # Scale random distribution using low_coh_thresh
                scale_num = np.sum(Na[:low_coh_thresh])
                scale_den = np.sum(Nr[:low_coh_thresh])
                Nr_scaled = Nr * (scale_num / scale_den)

                Na_safe = Na.copy()
                Na_safe[Na_safe == 0] = 1
                Prand = Nr_scaled / Na_safe
                Prand[:low_coh_thresh] = 1
                if Nr_max_nz_ix + 1 < len(Prand):
                    Prand[Nr_max_nz_ix + 1 :] = 0
                Prand[Prand > 1] = 1

                # Quick smoothing like filter(gausswin(7),1,...) in MATLAB
                gw = np.hanning(7)
                gw = gw / np.sum(gw)
                padded = np.concatenate((np.ones(7), Prand))
                smoothed = np.convolve(padded, gw, mode='full')
                smoothed = smoothed[6 : 6 + len(Prand)]  # like filter in MATLAB
                Prand_smooth = smoothed

                # Upsample by ~10
                x_old = np.arange(len(Prand_smooth))
                x_new = np.linspace(0, len(Prand_smooth) - 1, 10 * len(Prand_smooth))
                f_interp = interp1d(x_old, Prand_smooth, kind='linear')
                Prand_interp = f_interp(x_new)
                # Discard last 9 samples to mimic the MATLAB shape
                if len(Prand_interp) > len(x_new) - 9:
                    Prand_interp = Prand_interp[: (len(x_new) - 9)]

                weighting_new = np.zeros(n_ps_int, dtype=np.float64)
                for i_ps in range(n_ps_int):
                    cc = coh_ps[i_ps]
                    idx_lookup = int(round(cc * 1000))
                    idx_lookup = max(0, min(idx_lookup, len(Prand_interp) - 1))
                    val = Prand_interp[idx_lookup]
                    weighting_new[i_ps] = (1.0 - val) ** 2

                weighting = weighting_new
            else:
                # Weighted by SNR approach
                weighting_new = np.zeros(n_ps_int, dtype=np.float64)
                for i_ps in range(n_ps_int):
                    A_ps = A[i_ps, :]
                    ph_res_ps = ph_res[i_ps, :]  # angles
                    g_val = np.mean(A_ps * np.cos(ph_res_ps))
                    sigma_val = np.sqrt(0.5 * (np.mean(A_ps ** 2) - g_val ** 2))
                    if sigma_val != 0:
                        weighting_new[i_ps] = g_val / sigma_val
                    else:
                        weighting_new[i_ps] = 0.0
                weighting = weighting_new

        # Save partial results in each iteration
        save_h5(
            patch_dir,
            pmname,
            **{
            'ph_patch':ph_patch,
            'K_ps':K_ps,
            'C_ps':C_ps,
            'coh_ps':coh_ps,
            'N_opt':N_opt, 
            'ph_res':ph_res, 
            'step_number':step_number,
            'ph_grid':ph_grid,
            'n_trial_wraps':n_trial_wraps,
            'grid_ij':grid_ij,
            'grid_size':grid_size,
            'low_pass':low_pass,
            'i_loop':i_loop,
            'ph_weight':ph_weight,
            'Nr':Nr_scaled, # wrong, comes int but is float on matlab
            'Nr_max_nz_ix':Nr_max_nz_ix,
            'coh_bins':coh_bins,
            'gamma_change_save':gamma_change_save
            }
        )

    # Just like MATLAB, after while loop ends => logit(1)
    # End of function

def compute_coh_rand_with_pytorch(rand_ifg, bperp, n_trial_wraps):
    """
    Compute coherence using PyTorch with true batch processing.
    """
    # Convert inputs to PyTorch tensors
    rand_ifg_tensor = torch.tensor(rand_ifg, dtype=torch.float64)
    bperp_tensor = torch.tensor(bperp, dtype=torch.float64)

    # Move tensors to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rand_ifg_tensor = rand_ifg_tensor.to(device)
    bperp_tensor = bperp_tensor.to(device)

    # Convert all random phases to complex exponentials at once
    # Shape: [n_rand, n_ifg]
    exp_rand_ifg = torch.exp(1j * rand_ifg_tensor)

    # Process in batches to avoid memory issues
    batch_size = 1000  # Adjust based on your GPU memory
    n_rand = rand_ifg_tensor.size(0)
    coh_rand_tensor = torch.zeros(n_rand, dtype=torch.float64, device=device)

    for batch_start in range(0, n_rand, batch_size):
        batch_end = min(batch_start + batch_size, n_rand)
        batch_exp_rand_ifg = exp_rand_ifg[batch_start:batch_end]
        
        # Expand bperp for batch processing
        # Shape: [batch_size, n_ifg]
        batch_bperp = bperp_tensor.expand(batch_end - batch_start, -1)
        
        # Process entire batch at once
        _, _, coh_batch, _ = ps_topofit_torch(
            batch_exp_rand_ifg,  # Shape: [batch_size, n_ifg]
            batch_bperp,         # Shape: [batch_size, n_ifg]
            n_trial_wraps,
            'n'
        )
        
        coh_rand_tensor[batch_start:batch_end] = coh_batch

    return coh_rand_tensor.cpu().numpy()
