import numpy as np
import os
from ..misc import get_module_info
from ..logger import appLogger
from .utils import read_h5,save_h5,gaussian1D
from .ps_topofit import ps_topofit
from .clap_filt import clap_filt
from tqdm import tqdm
from scipy import signal

def step_2_ps_estmgm(workdir:str,patch:str,parms:dict) -> None:
    """
    Estimating gamma for candidate pixels

    Args:
        workdir:str - path to the working directory
        patch:str - patch currently being processed
        parms:dict - parameters from parms.json
    """
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {} {}".format(
            get_module_info(),workdir, patch)
    )
    patch_dir = os.path.join(workdir,patch)

    # Constants
    rho = 830000  # mean range - only approximate
    n_rand = 300000  # number of simulated random-phase pixels
    low_coh_thresh = 31  # equivalent to coherence = 0.31
    coh_bins = np.arange(0.005, 1.0, 0.01)

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
    psver = int(read_h5(os.path.join(patch_dir, 'psver.h5'))['psver'])

    # Build filenames
    psname = f'ps{int(psver)}.h5'
    phname = f'ph{int(psver)}.h5'
    bpname = f'bp{int(psver)}.h5'
    laname = f'la{int(psver)}.h5'
    pmname = f'pm{int(psver)}.h5'
    daname = f'da{int(psver)}.h5'

    # Load PS and baseline perpendicular data
    ps_data = read_h5(os.path.join(patch_dir, psname))
    bp_data = read_h5(os.path.join(patch_dir,bpname))

    if os.path.exists(os.path.join(patch_dir,daname)):
        da=read_h5(os.path.join(patch_dir,daname))
        D_A=da['D_A'];
    else:
        D_A=np.ones(ps_data['n_ps'],1);


    if os.path.exists(os.path.join(patch_dir,phname)):
        phin=read_h5(os.path.join(patch_dir,phname))
        ph=phin['ph']
    else:
        ph=ps_data['ph']

    # Exclude master image from ph and bperp
    master_ix = int(ps_data['master_ix'])
    ph = np.delete(ph, master_ix, axis=1)
    bperp = np.delete(ps_data['bperp'], master_ix)
    n_ifg = int(ps_data['n_ifg'])-1
    n_ps = int(ps_data['n_ps'])
    xy = ps_data['xy']

    # Normalize ph
    A = np.abs(ph).astype(np.float32)
    A[A == 0] = 1.0
    ph = ph / A

    # ===============================================
    # The code below needs to be made sensor specific
    # ===============================================
    # load look angle
    la_data = read_h5(os.path.join(patch_dir,laname))
    inc_mean = np.mean(la_data['la']) + 0.052
    max_K = max_topo_err / (wavelength * rho * np.sin(inc_mean) / 4 / np.pi)
    # ===============================================
    # The code below needs to be made sensor specific
    # ===============================================

    # Max K and approximate number of trial wraps
    bperp_range = float(np.max(bperp) - np.min(bperp))
    max_K = max_topo_err / (wavelength * rho * np.sin(inc_mean) / (4.0 * np.pi))
    n_trial_wraps = bperp_range * max_K / (2.0 * np.pi)
    print(f'n_trial_wraps={n_trial_wraps}')

    print('Initialising random distribution...')
    np.random.seed(2005)
    rand_ifg = (2.0 * np.pi * np.random.rand(n_ifg, n_rand)).T
    coh_rand = np.zeros(n_rand, dtype=np.float32)
    for i in tqdm(range(n_rand),desc="Running ps_topofit"):
        K_r, C_r, coh_r, phase_residual = ps_topofit(np.exp(1j * rand_ifg[i, :]), bperp, n_trial_wraps)
        coh_rand[i] = coh_r

    bin_width = 0.01
    bin_edges = np.append(coh_bins - bin_width/2, coh_bins[-1] + bin_width/2)
    Nr, _ = np.histogram(coh_rand, bins=bin_edges)
    i = len(Nr) - 1
    while Nr[i] == 0:
        i = i - 1
    Nr_max_nz_ix = i

    # Calculate grid indices
    grid_ij = np.zeros((n_ps, 2), dtype=int)
    grid_ij[:, 0] = np.floor((xy[:, 2] - np.min(xy[:, 2]) + 1e-6) / grid_size)
    grid_ij[grid_ij[:, 0] == np.max(grid_ij[:, 0]), 0] = np.max(grid_ij[:, 0]) - 1

    grid_ij[:, 1] = np.floor((xy[:, 1] - np.min(xy[:, 1]) + 1e-6) / grid_size)
    grid_ij[grid_ij[:, 1] == np.max(grid_ij[:, 1]), 1] = np.max(grid_ij[:, 1]) - 1

    # Assuming grid_ij is a NumPy array with at least two columns
    n_i = np.max(grid_ij[:, 0]) + 1
    n_j = np.max(grid_ij[:, 1]) + 1

    print(f'{n_ps} PS candidates to process')

    K_ps = np.zeros((n_ps, 1))
    C_ps = np.zeros((n_ps, 1))
    coh_ps = np.zeros((n_ps, 1))
    N_opt = np.zeros((n_ps, 1))
    ph_res = np.zeros((n_ps, n_ifg), dtype=np.float32)
    ph_patch = np.zeros(ph.shape, dtype=np.complex64)
    coh_ps_save = np.zeros((n_ps, 1))

    # --------------------
    # Main iteration loop
    # --------------------
    weighting = 1. /  D_A
    step_number = 1
    loop_end_sw = 0
    i_loop = 1
    gamma_change_save = 0

    while loop_end_sw==0:
        print(f'iteration ##### {i_loop}')
        print('Calculating patch phases...')

        ph_grid = np.zeros((n_i, n_j, n_ifg), dtype=np.complex64)
        ph_filt = np.copy(ph_grid)
        K_ps_col = K_ps.reshape(-1, 1)
        weighting_col = weighting.reshape(-1, 1)
        ph_weight = ph * np.exp(-1j * bp_data["bperp_mat"] * K_ps_col) * weighting_col

        for i in range(n_ps):
            row = grid_ij[i, 0]
            col = grid_ij[i, 1]
            ph_grid[row, col, :] += ph_weight[i, :]

        for i in range(n_ifg):
            ph_filt[:, :, i] = clap_filt(
                ph_grid[:, :, i], 
                clap_alpha, 
                clap_beta, 
                int(n_win * 0.75), 
                int(n_win * 0.25), 
                low_pass)
        
        for i in range(n_ps):
            row = grid_ij[i, 0]
            col = grid_ij[i, 1]
            ph_patch[i, :] = ph_filt[row, col, :]
        
        ix = ph_patch != 0
        ph_patch[ix] = ph_patch[ix] / np.abs(ph_patch[ix])
        
        # #########################################################
        # Estimating topo error
        # #########################################################

        for i in range(n_ps):
            psdph = ph[i,:] * np.conj(ph_patch[i,:])
            if np.sum(psdph == 0) == 0:
                # Assuming bp.bperp_mat is a 2D array
                Kopt, Copt, cohopt, ph_residual = ps_topofit(psdph, bp_data["bperp_mat"][i,:].T, n_trial_wraps)
                K_ps[i] = Kopt[0]
                C_ps[i] = Copt
                coh_ps[i] = cohopt
                N_opt[i] = len(Kopt)
                ph_res[i,:] = np.angle(ph_residual)
            else:
                K_ps[i] = np.nan
                coh_ps[i] = 0
            if (i + 1) % 100000 == 0:
                print(f'{i+1} PS processed')
        
        gamma_change_rms = np.sqrt(np.sum((coh_ps - coh_ps_save) ** 2) / n_ps)
        gamma_change_change = gamma_change_rms - gamma_change_save
        print(f'gamma_change_change={gamma_change_change}')
        gamma_change_save = gamma_change_rms
        coh_ps_save = coh_ps

        # #########################################################
        # End of topo error estimation
        # #########################################################

        if abs(gamma_change_change) < gamma_change_convergence or i_loop >= gamma_max_iterations:
            loop_end_sw = 1
        else:
            i_loop = i_loop + 1
            if filter_weighting.lower() == 'p-square':
                bin_width = 0.01
                bin_edges = np.append(coh_bins - bin_width/2, coh_bins[-1] + bin_width/2)
                Na, _ = np.histogram(coh_ps, bins=bin_edges)
                Nr = Nr * sum(Na[:low_coh_thresh]) / sum(Nr[:low_coh_thresh])  # scale random distribution
                Na[Na == 0] = 1  # avoid divide by zero

                Prand = Nr / Na
                Prand[:low_coh_thresh] = 1
                Prand[Nr_max_nz_ix+1:] = 0
                Prand[Prand > 1] = 1
                
                window_len = 7
                gauss_win = gaussian1D(7)
                padded = np.concatenate([np.ones(window_len), Prand])
                filtered = signal.lfilter(gauss_win, 1, padded)
                Prand = filtered / np.sum(gauss_win)
                Prand = Prand[window_len:]

                R = 10
                fp = np.concatenate(([1.0], Prand))
                N = len(fp)
                xp = np.arange(N, dtype=np.float64)
                num_points = (N - 1) * R + 1
                x = np.linspace(xp[0], xp[-1],num=num_points)
                Prand = np.interp(x, xp, fp)

                indices = np.round(coh_ps * 1000).astype(int) + 1
                indices = np.clip(indices, 0, len(Prand) - 1)
                Prand_ps = Prand[indices]
                weighting = (1 - Prand_ps) ** 2
            else:
                g = np.mean(A * np.cos(ph_res), axis=1)
                sigma_n = np.sqrt(0.5 * (np.mean(A**2, axis=1) - g**2))
                weighting = np.zeros_like(sigma_n)
                non_zero_mask = (sigma_n != 0)
                weighting[non_zero_mask] = g[non_zero_mask] / sigma_n[non_zero_mask]  # snr
    
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
                'Nr':Nr,
                'Nr_max_nz_ix':Nr_max_nz_ix,
                'coh_bins':coh_bins,
                'gamma_change_save':gamma_change_save
                }
            )
