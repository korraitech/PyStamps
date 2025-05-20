import numpy as np
from scipy import sparse
import os
from .utils import read_h5, save_h5

def uw_space_time(workdir, day, ifgday_ix, bperp, time_win, n_trial_wraps):
    print('Unwrapping in time-space...')

    # Load data
    uw = read_h5(os.path.join(workdir,'uw_grid.h5'))
    ui = read_h5(os.path.join(workdir,'uw_interp.h5'))

    n_ifg = uw['n_ifg']
    n_edge = ui['n_edge']
    n_image = day.shape[0]
    
    # Calculate dph_space
    dph_space = uw['ph'][ui['edgs'][:, 2]-1, :] * np.conj(uw['ph'][ui['edgs'][:, 1]-1, :])

    dph_space = dph_space / np.abs(dph_space)

    # Create G matrix
    G = np.zeros((n_ifg, n_image))
    for i in range(n_ifg):
        G[i, ifgday_ix[i, 0]] = -1
        G[i, ifgday_ix[i, 1]] = 1

    # Find non-zero columns
    nzc_ix = np.sum(np.abs(G), axis=0) != 0
    day = day[nzc_ix]
    n_image = day.shape[0]
    G = G[:, nzc_ix]
    zc_ix = np.where(nzc_ix == 0)[0]
    zc_ix = np.sort(zc_ix)[::-1]  # Sort in descending order

    for i in range(len(zc_ix)):
        ifgday_ix[ifgday_ix > zc_ix[i]] -= 1

    print(f'   Estimating look angle error')
    bperp_range = np.max(bperp) - np.min(bperp)
    ix = np.where(np.abs(np.diff(ifgday_ix, axis=1)) == 1)[0]

    if len(ix) >= len(day) - 1:
        print('   using sequential daisy chain of interferograms')
        dph_sub = dph_space[:, ix]
        bperp_sub = bperp[ix]
        bperp_range_sub = np.max(bperp_sub) - np.min(bperp_sub)
        n_trial_wraps = n_trial_wraps * (bperp_range_sub / bperp_range)
    else:
        ifgs_per_image = np.sum(np.abs(G), axis=0)
        max_ifgs_per_image = np.max(ifgs_per_image)
        max_ix = np.argmax(ifgs_per_image)

        if max_ifgs_per_image >= len(day) - 2:
            print('   Using sequential daisy chain of interferograms')
            ix = G[:, max_ix] != 0
            gsub = G[ix, max_ix]
            sign_ix = -np.sign(gsub.astype(np.float32))
            dph_sub = dph_space[:, ix]
            bperp_sub = bperp[ix]
            bperp_sub[sign_ix == -1] = -bperp_sub[sign_ix == -1]
            bperp_sub = np.append(bperp_sub, 0)
            sign_ix = np.tile(sign_ix, (n_edge, 1))
            dph_sub[sign_ix == -1] = np.conj(dph_sub[sign_ix == -1])
            dph_sub = np.column_stack((dph_sub, np.mean(np.abs(dph_sub), axis=1)))
            slave_ix = np.sum(ifgday_ix[ix, :], axis=1) - max_ix
            day_sub = day[np.append(slave_ix, max_ix)]
            sort_ix = np.argsort(day_sub)
            day_sub = day_sub[sort_ix]
            dph_sub = dph_sub[:, sort_ix]
            bperp_sub = bperp_sub[sort_ix]
            bperp_sub = np.diff(bperp_sub)
            bperp_range_sub = np.max(bperp_sub) - np.min(bperp_sub)
            n_trial_wraps = n_trial_wraps * (bperp_range_sub / bperp_range)
            n_sub = len(day_sub)
            dph_sub = dph_sub[:, 1:] * np.conj(dph_sub[:, :-1])
            dph_sub = dph_sub / np.abs(dph_sub)
        else:
            dph_sub = dph_space
            bperp_sub = bperp
            bperp_range_sub = bperp_range

    trial_mult = np.arange(-np.ceil(8 * n_trial_wraps), np.ceil(8 * n_trial_wraps) + 1)
    n_trials = len(trial_mult)
    trial_phase = bperp_sub / bperp_range_sub * np.pi / 4
    trial_phase_mat = np.exp(-1j * trial_phase[:, np.newaxis] * trial_mult)
    
    K = np.zeros(n_edge, dtype=np.float32)
    coh = np.zeros(n_edge, dtype=np.float32)

    for i in range(n_edge):
        cpxphase = dph_sub[i, :].T
        cpxphase_mat = np.tile(cpxphase[:, np.newaxis], (1, n_trials))
        phaser = trial_phase_mat * cpxphase_mat
        phaser_sum = np.sum(phaser, axis=0)
        coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))
        
        coh_max = np.max(coh_trial)
        coh_max_ix = np.argmax(coh_trial)
        
        falling_ix = np.where(np.diff(coh_trial[:coh_max_ix+1]) < 0)[0]
        if len(falling_ix) > 0:
            peak_start_ix = falling_ix[-1] + 1
        else:
            peak_start_ix = 0
            
        rising_ix = np.where(np.diff(coh_trial[coh_max_ix:]) > 0)[0]
        if len(rising_ix) > 0:
            peak_end_ix = rising_ix[0] + coh_max_ix
        else:
            peak_end_ix = n_trials - 1
            
        coh_trial[peak_start_ix:peak_end_ix+1] = 0
        
        if coh_max - np.max(coh_trial) > 0.1:
            K0 = np.pi / 4 / bperp_range_sub * trial_mult[coh_max_ix]
            resphase = cpxphase * np.exp(-1j * (K0 * bperp_sub))
            offset_phase = np.sum(resphase)
            resphase = np.angle(resphase * np.conj(offset_phase))
            weighting = np.abs(cpxphase)
            mopt = np.linalg.lstsq(weighting[:, np.newaxis] * bperp_sub[:, np.newaxis], 
                                    weighting * resphase, rcond=None)[0]
            K[i] = K0 + mopt[0]
            phase_residual = cpxphase * np.exp(-1j * (K[i] * bperp_sub))
            mean_phase_residual = np.sum(phase_residual)
            coh[i] = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))

    K[coh < 0.31] = 0
    dph_space = dph_space * np.exp(-1j * K[:, np.newaxis] * bperp)

    spread = sparse.csr_matrix((n_edge, n_ifg))

    print(f'   Smoothing ....')
    dph_smooth_ifg = np.full(dph_space.shape, np.nan, dtype=np.float32)

    for i in range(n_image):
        ix = G[:, i] != 0
        if np.sum(ix) >= n_image - 2:
            gsub = G[ix, i]
            dph_sub = dph_space[:, ix]
            sign_ix = np.tile(-np.sign(gsub.astype(np.float32)), (n_edge, 1))
            dph_sub[sign_ix == -1] = np.conj(dph_sub[sign_ix == -1])
            slave_ix = np.sum(ifgday_ix[ix, :], axis=1) - i
            day_sub = day[slave_ix]
            sort_ix = np.argsort(day_sub)
            day_sub = day_sub[sort_ix]
            dph_sub = dph_sub[:, sort_ix]
            dph_sub_angle = np.angle(dph_sub)
            n_sub = len(day_sub)
            dph_smooth = np.zeros((n_edge, n_sub), dtype=np.float32)

            for i1 in range(n_sub):
                time_diff = (day_sub[i1] - day_sub).T
                weight_factor = np.exp(-(time_diff ** 2) / 2 / time_win ** 2)
                weight_factor = weight_factor / np.sum(weight_factor)

                dph_mean = np.sum(dph_sub * np.tile(weight_factor, (n_edge, 1)), axis=1)
                dph_mean_adj = np.mod(dph_sub_angle - np.tile(np.angle(dph_mean), 
                                                              (n_sub, 1)).T + np.pi, 2 * np.pi) - np.pi
                
                GG = np.column_stack((np.ones(n_sub), time_diff))
                if GG.shape[0] > 1:
                    m = np.linalg.lstsq(GG * weight_factor[:, np.newaxis], 
                                      dph_mean_adj.T * weight_factor, rcond=None)[0]
                else:
                    m = np.zeros((GG.shape[1], n_edge))

                dph_smooth[:, i1] = dph_mean * np.exp(1j * m[0, :])

            dph_smooth_sub = np.cumsum(np.column_stack((
                np.angle(dph_smooth[:, 0]),
                np.angle(dph_smooth[:, 1:] * np.conj(dph_smooth[:, :-1]))
            )), axis=1)

            close_master_ix = np.where(slave_ix - i > 0)[0]
            if len(close_master_ix) == 0:
                close_master_ix = np.array([n_sub - 1])
            else:
                close_master_ix = close_master_ix[0]
                if close_master_ix > 0:
                    close_master_ix = np.array([close_master_ix - 1, close_master_ix])

            dph_close_master = np.mean(dph_smooth_sub[:, close_master_ix], axis=1)
            dph_smooth_sub = dph_smooth_sub - np.tile(dph_close_master - np.angle(np.exp(1j * dph_close_master)), (n_sub, 1)).T
            dph_smooth_sub = dph_smooth_sub * sign_ix

            already_sub_ix = np.where(~np.isnan(dph_smooth_ifg[0, ix]))[0]
            ix = np.where(ix)[0]
            already_ix = ix[already_sub_ix]
            
            std_noise1 = np.std(np.angle(dph_space[:, already_ix] * np.exp(-1j * dph_smooth_ifg[:, already_ix])))
            std_noise2 = np.std(np.angle(dph_space[:, already_ix] * np.exp(-1j * dph_smooth_sub[:, already_sub_ix])))
            
            keep_ix = np.ones(n_sub, dtype=bool)
            keep_ix[already_sub_ix[std_noise1 < std_noise2]] = False
            
            dph_smooth_ifg[:, ix[keep_ix]] = dph_smooth_sub[:, keep_ix]

    dph_noise = np.angle(dph_space * np.exp(-1j * dph_smooth_ifg))
    dph_noise[np.std(dph_noise, axis=1) > 1.2, :] = np.nan
    dph_space_uw = dph_smooth_ifg + dph_noise
    dph_space_uw = dph_space_uw + K[:, np.newaxis] * bperp
    
    # Save results
    save_h5(workdir, 'uw_space_time.h5', **{'dph_space_uw':dph_space_uw,
                                            'dph_noise':dph_noise,'G':G,
                                            'spread':spread}) 
    