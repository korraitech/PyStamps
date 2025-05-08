import os
import numpy as np
from scipy import signal
from scipy import fft
from .utils import read_h5,save_h5
from .ps_topofit import ps_topofit

def calculate_threshold(D_A,D_A_max,D_A_mean,pm,Nr_dist,
                        select_method,max_percent_rand,min_coh,low_coh_thresh) ->float:
    for i in range(len(D_A_max)-1):
        mask = (D_A > D_A_max[i]) & (D_A <= D_A_max[i+1])
        coh_chunk = pm['coh_ps'][mask]
        D_A_mean[i] = np.mean(D_A[mask])
        coh_chunk = coh_chunk[coh_chunk != 0]

        bin_width = 0.01
        coh_bins = pm['coh_bins']
        bin_edges = np.append(coh_bins - bin_width/2, coh_bins[-1] + bin_width/2)
        Na, _ = np.histogram(coh_chunk, bins=bin_edges)

        Nr = Nr_dist * np.sum(Na[:low_coh_thresh]) / np.sum(Nr_dist[:low_coh_thresh])
        Na[Na == 0] = 1
        
        if select_method.upper() == 'PERCENT':
            percent_rand = np.flip(np.cumsum(np.flip(Nr))) / np.cumsum(np.flip(Na)) * 100
        else:
            percent_rand = np.flip(np.cumsum(np.flip(Nr)))  # absolute number
        
        ok_ix = np.where(percent_rand < max_percent_rand)[0]
        if len(ok_ix) == 0:
            min_coh[i] = 1  # no threshold meets criteria
        else:
            min_fit_ix = np.min(ok_ix) - 3
            if min_fit_ix <= 0:
                min_coh[i] = np.nan
            else:
                max_fit_ix = np.min(ok_ix) + 2
                if max_fit_ix > 100:
                    max_fit_ix = 100
                
                x_vals = percent_rand[min_fit_ix:max_fit_ix+1]
                y_vals = np.arange(min_fit_ix * 0.01, (max_fit_ix + 1) * 0.01, 0.01)
                
                p = np.polyfit(x_vals, y_vals, 3)
                min_coh[i] = np.polyval(p, max_percent_rand)

    nonnanix = ~np.isnan(min_coh)
    coh_thresh_coeffs = []
    if np.sum(nonnanix) < 1:
        print('Not enough random phase pixels to set gamma threshold')
        coh_thresh = 0.3
    else:
        min_coh_filtered = min_coh[nonnanix]
        D_A_mean_filtered = D_A_mean[nonnanix]

        if min_coh_filtered.size > 1:
            coh_thresh_coeffs = np.polyfit(D_A_mean_filtered, min_coh_filtered, 1)
            if coh_thresh_coeffs[0] > 0:
                coh_thresh = np.polyval(coh_thresh_coeffs, D_A)
            else:
                coh_thresh = np.polyval(coh_thresh_coeffs, 0.35)
                coh_thresh_coeffs = []
        else:
            coh_thresh = min_coh_filtered
            coh_thresh_coeffs = []
    
    return np.maximum(coh_thresh, 0), coh_thresh_coeffs

def clap_filt_patch(ph,alpha,beta,low_pass) -> np.ndarray:
    ph[np.isnan(ph)] = 0
    std_dev = (7 - 1) / (2 * 2.5)
    gauss_win_1d = signal.windows.gaussian(7, std=std_dev, sym=True)
    B = np.outer(gauss_win_1d, gauss_win_1d)
    ph_fft = fft.fft2(ph)
    
    H = np.abs(ph_fft)
    H = fft.ifftshift(signal.convolve2d(fft.fftshift(H), B, mode='same'))
    meanH = np.median(H.flatten())
    if meanH != 0:
        H = H / meanH
    H = H ** alpha
    H = H - 1
    H[H < 0] = 0
    G = H * beta + low_pass
    return fft.ifft2(ph_fft * G)

def step_3_ps_select(workdir:str,patch:str,parms:dict) -> None:
    """
    Select PS based on gamma and D_A, re-estimating coherence if requested.
    """
    print("Running Step-03 ...\t[{}]".format(patch))
    print('Selecting stable-phase pixels...')
    patch_dir = os.path.join(workdir,patch)

    slc_osf = int(parms['slc_osf'])
    clap_alpha = int(parms['clap_alpha'])
    clap_beta = float(parms['clap_beta'])
    n_win = int(parms['clap_win'])
    select_method = parms['select_method']
    max_percent_rand = 0.0
    max_density_rand = 0.0
    if select_method.upper() == 'PERCENT':
        max_percent_rand = float(parms['percent_rand'])
    else:
        max_density_rand = float(parms['density_rand'])
    gamma_stdev_reject = float(parms['gamma_stdev_reject'])
    drop_ifg_index = np.array(parms['drop_ifg_index'])
    low_coh_thresh = 31

    psver = int(read_h5(os.path.join(patch_dir, 'psver.h5'))['psver'])
    psname = f'ps{psver}.h5'
    phname = f'ph{psver}.h5'
    pmname = f'pm{psver}.h5'
    daname = f'da{psver}.h5'
    bpname = f'bp{psver}.h5'
    selectname = f'select{psver}.h5'
    
    ps = read_h5(os.path.join(patch_dir, psname))
    n_ifg = int(ps['n_ifg'])
    ifg_index = np.setdiff1d(np.arange(0, n_ifg ), drop_ifg_index)

    if os.path.exists(os.path.join(patch_dir, phname)):
        phin = read_h5(os.path.join(patch_dir, phname))
        ph = phin['ph']
    else:
        ph = ps.ph
    
    bperp = ps['bperp']
    master_ix = int(ps['master_ix'])
    no_master_ix = np.setdiff1d(np.arange(0, n_ifg), [master_ix])
    ifg_index = np.setdiff1d(ifg_index, [master_ix])
    ifg_index[ifg_index > master_ix] = ifg_index[ifg_index > master_ix] - 1
    ph = ph[:, no_master_ix]
    bperp = bperp[no_master_ix]
    n_ifg = len(no_master_ix)
    n_ps = int(ps['n_ps'])
    xy = ps['xy']

    pm = read_h5(os.path.join(patch_dir, pmname))
    if os.path.exists(os.path.join(patch_dir, daname)):
        da = read_h5(os.path.join(patch_dir, daname))
        D_A = da['D_A']
    else:
        D_A = np.ones(pm['coh_ps'].shape)
    
    if len(D_A) > 0 and D_A.shape[0] >= 10000:
        D_A_sort = np.sort(D_A)
        if D_A.shape[0] >= 50000:
            bin_size = 10000
        else:
            bin_size = 2000
        
        indices = np.arange(bin_size-1, len(D_A_sort)-bin_size, bin_size)
        D_A_max = np.concatenate(([0], D_A_sort[indices], [D_A_sort[-1]]))
    else:
        D_A_max = np.array([0, 1])
        D_A = np.ones_like(pm['coh_ps'])
    
    if not select_method.upper() == 'PERCENT':
        patch_area = np.prod(np.max(xy[:, 1:3], axis=0) - np.min(xy[:, 1:3], axis=0)) / 1e6
        max_percent_rand = max_density_rand * patch_area / (len(D_A_max) - 1)

    min_coh = np.zeros(len(D_A_max) - 1)
    D_A_mean = np.zeros(D_A_max.shape[0] - 1)
    Nr_dist = pm['Nr']

    coh_thresh,coh_thresh_coeffs = calculate_threshold(D_A,D_A_max,D_A_mean,pm,Nr_dist,
                    select_method,max_percent_rand,min_coh,low_coh_thresh)
    
    print(f'Initial gamma threshold: {np.min(coh_thresh):.3f} at D_A={np.min(D_A):.2f} to {np.max(coh_thresh):.3f} at D_A={np.max(D_A):.2f}')

    ix = np.where(pm['coh_ps'] > coh_thresh)[0]
    n_ps = len(ix)
    print(f'{n_ps} PS selected initially')

    # Above code are validated
    ###########################################################################
    #### reject part-time PS
    ###########################################################################

    if gamma_stdev_reject > 0:
        ph_res_cpx = np.exp(1j * pm['ph_res'][:, ifg_index])
        coh_std = np.zeros(len(ix))
        ph_magnitude = lambda ph: abs(np.sum(ph))/len(ph)
        for i in range(len(ix)):
            coh_std[i] = np.std([ph_magnitude(
                np.random.choice(ph_res_cpx[ix[i], ifg_index], 
                                 size=len(ph_res_cpx[ix[i], ifg_index]), 
                                 replace=True)) for _ in range(100)])
        ix = ix[coh_std < gamma_stdev_reject]
        n_ps = len(ix)
        print(f'{n_ps} PS left after pps rejection')
    
    del pm['ph_res']
    del pm['ph_patch']
    ph_patch2 = np.zeros((n_ps, n_ifg), dtype=np.complex64)
    ph_res2 = np.zeros((n_ps, n_ifg), dtype=np.float64)
    ph = ph[ix, :]
    
    if len(np.atleast_1d(coh_thresh)) > 1:
        coh_thresh = coh_thresh[ix]
    
    n_i = np.max(pm['grid_ij'][:, 0]) + 1
    n_j = np.max(pm['grid_ij'][:, 1]) + 1
    K_ps2 = np.zeros(n_ps, dtype=np.float64)
    C_ps2 = np.zeros(n_ps, dtype=np.float64)
    coh_ps2 = np.zeros(n_ps, dtype=np.float64)
    ph_filt = np.zeros((n_win, n_win, n_ifg), dtype=np.complex64)

    for i in range(n_ps):
        ps_ij = pm['grid_ij'][ix[i], :]
        i_min = max(ps_ij[0] - n_win//2, 0)
        i_max = i_min + n_win - 1
        if i_max >= n_i:
            i_min = i_min - (i_max - n_i + 1)
            i_max = n_i - 1

        j_min = max(ps_ij[1] - n_win//2, 0)
        j_max = j_min + n_win - 1
        if j_max >= n_j:
            j_min = j_min - (j_max - n_j + 1)
            j_max = n_j - 1

        # it could occur that your patch size is smaller than the filter size
        # crude bug fix is to drop this patch. It needs fixing in future...
        if j_min < 0 or i_min < 0:
            # THIS NEEDS TO BECOME AN ACTUAL FIX, but not sure how...
            ph_patch2[i, :] = 0.0
        else:
            # remove the pixel for which the smoothign is computed
            ps_bit_i = ps_ij[0] - i_min
            ps_bit_j = ps_ij[1] - j_min
            ph_bit = pm['ph_grid'][i_min:i_max+1, j_min:j_max+1, :].copy()
            ph_bit[ps_bit_i, ps_bit_j, :] = 0
            
            # JJS oversample update for PS removal + [MA] general usage update
            ix_i = np.arange(ps_bit_i-(slc_osf-1), ps_bit_i+(slc_osf-1)+1)
            ix_i = ix_i[(ix_i >= 0) & (ix_i < ph_bit.shape[0])]
            ix_j = np.arange(ps_bit_j-(slc_osf-1), ps_bit_j+(slc_osf-1)+1)
            ix_j = ix_j[(ix_j >= 0) & (ix_j < ph_bit.shape[1])]
            ph_bit[np.ix_(ix_i, ix_j)] = 0

            for i_ifg in range(n_ifg):
                ph_filt[:, :, i_ifg] = clap_filt_patch(ph_bit[:, :, i_ifg], clap_alpha, clap_beta, pm['low_pass'])
            ph_patch2[i, :] = ph_filt[ps_bit_i, ps_bit_j, :]

        if i % 10000 == 9999:
                print(f"{i+1} patches re-estimated")

    del pm['ph_grid']
    bp = read_h5(os.path.join(patch_dir, bpname))
    bperp_mat = bp['bperp_mat'][ix, :]

    for i in range(n_ps):
        psdph = ph[i, :] * np.conj(ph_patch2[i, :])
        if np.sum(psdph == 0) == 0:  # insist on a non-null value in every ifg
            psdph = psdph / np.abs(psdph)
            [Kopt, Copt, cohopt, ph_residual] = ps_topofit(psdph[ifg_index], bperp_mat[i, ifg_index].T, pm['n_trial_wraps'])
            K_ps2[i] = Kopt[0]
            C_ps2[i] = Copt
            coh_ps2[i] = cohopt
            ph_res2[i, ifg_index] = np.angle(ph_residual)
        else:
            K_ps2[i] = np.nan
            coh_ps2[i] = np.nan
        
        if i % 10000 == 0 and i > 0:
            print(f'{i} coherences re-estimated')

    pm['coh_ps'][ix] = coh_ps2.reshape(-1, 1)
    
    coh_thresh,coh_thresh_coeffs = calculate_threshold(D_A,D_A_max,D_A_mean,pm,Nr_dist,
                    select_method,max_percent_rand,min_coh,low_coh_thresh)

    print(f"Reestimation gamma threshold: {np.min(coh_thresh):.3f} at D_A={np.min(D_A):.2f} to {np.max(coh_thresh):.3f} at D_A={np.max(D_A):.2f}")

    bperp_range = max(bperp) - min(bperp)
    keep_ix = (coh_ps2 > coh_thresh) & (abs(pm["K_ps"][ix].flatten() - K_ps2) < 2*np.pi/bperp_range)

    print(f"{sum(keep_ix)} ps selected after re-estimation of coherence")
    
    if np.sum(keep_ix) == 0:
        print("***No PS points left. Updating the stamps log for this****")
    
    save_h5(patch_dir, 
            selectname, 
            **{
                'ix':ix,
                'keep_ix':keep_ix,
                'ph_patch2':ph_patch2,
                'ph_res2':ph_res2,
                'K_ps2':K_ps2,
                'C_ps2':C_ps2,
                'coh_ps2':coh_ps2,
                'coh_thresh':coh_thresh,
                'coh_thresh_coeffs':coh_thresh_coeffs,
                'clap_alpha':clap_alpha,
                'clap_beta':clap_beta,
                'n_win':n_win,
                'max_percent_rand':max_percent_rand,
                'gamma_stdev_reject':gamma_stdev_reject,
                'ifg_index':ifg_index
            }
    )
