import os
import numpy as np
from ..misc import get_module_info
from ..logger import appLogger
from .utils import read_h5,save_h5

def calculate_threshold(D_A,D_A_max,pm,Nr_dist,select_method,max_percent_rand,low_coh_thresh) -> tuple:
    min_coh = np.zeros(len(D_A_max) - 1)
    D_A_mean = np.zeros(D_A_max.shape[0] - 1)
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
                y_vals = np.arange(min_fit_ix, max_fit_ix + 1) * 0.01
                
                p = np.polyfit(x_vals, y_vals, 3)
                min_coh[i] = np.polyval(p, max_percent_rand)

    nonnanix = ~np.isnan(min_coh)
    coh_thresh_coeffs = []
    if np.sum(nonnanix) < 1:
        print('Not enough random phase pixels to set gamma threshold - using default threshold of 0.3')
        coh_thresh = np.full(pm['coh_ps'].shape[0], 0.3)
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
            coh_thresh = np.full(pm['coh_ps'].shape[0], min_coh_filtered)
            coh_thresh_coeffs = []
    
    coh_thresh[coh_thresh<0] = 0
    return (coh_thresh, coh_thresh_coeffs)

def step_3_ps_select(workdir:str,patch:str,parms:dict) -> None:
    """
    Select PS based on gamma and D_A, re-estimating coherence if requested.

    Args:
        workdir:str - path to the working directory
        patch:str - patch currently being processed
        parms:dict - parameters from parms.json
    """
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {} {}".format(
            get_module_info(),workdir, patch)
    )
    patch_dir = os.path.join(workdir,patch)

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
        D_A_max = np.concatenate([[0], D_A_sort[bin_size-1::bin_size][:-1],[D_A_sort[-1]]])
    else:
        D_A_max = np.array([0, 1])
        D_A = np.ones_like(pm['coh_ps'])
    
    if not select_method.upper() == 'PERCENT':
        patch_area = np.prod(np.max(xy[:, 1:3], axis=0) - np.min(xy[:, 1:3], axis=0)) / 1e6
        max_percent_rand = max_density_rand * patch_area / (len(D_A_max) - 1)

    Nr_dist = pm['Nr']

    coh_thresh,coh_thresh_coeffs = calculate_threshold(D_A,D_A_max,pm,Nr_dist,
                    select_method,max_percent_rand,low_coh_thresh)
    
    print(f'Initial gamma threshold: {np.min(coh_thresh):.3f} at D_A={np.min(D_A):.2f} to {np.max(coh_thresh):.3f} at D_A={np.max(D_A):.2f}')

    ix = []
    for i in range(len(coh_thresh)):
        if pm['coh_ps'][i] > coh_thresh[i]:
            ix.append(i)
    
    ix = np.array(ix)
    n_ps = len(ix)
    print(f'{n_ps} PS selected initially')

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
    
    del pm['ph_grid']
    ph_patch2 = pm['ph_patch'][ix, :]
    ph_res2 = pm['ph_res'][ix, :]
    K_ps2 = pm['K_ps'][ix]
    C_ps2 = pm['C_ps'][ix]
    coh_ps2 = pm['coh_ps'][ix]
    keep_ix = np.ones_like(ix, dtype=bool)

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
