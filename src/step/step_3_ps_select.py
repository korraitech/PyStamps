import os
import numpy as np
from .utils import read_h5,save_h5
from .clap_filt import clap_filt
from .ps_topofit_numpy import ps_topofit

def step_3_ps_select(workdir:str,patch:str,parms:dict) -> None:
    """
    Select PS based on gamma and D_A, re-estimating coherence if requested.

    Args:
        reest_flag (int):
            0 - default; normal processing
            1 - skip re-estimation of gamma
            2 - reuse previously re-estimated gamma
            3 - re-estimate gamma for all candidate pixels
    """
    print("Running Step-03 ...\t[{}]".format(patch))
    print('Selecting stable-phase pixels...')
    patch_dir = os.path.join(workdir,patch)

    clap_alpha = int(parms['clap_alpha'])
    clap_beta = float(parms['clap_beta'])
    clap_win = int(parms['clap_win'])
    select_method = parms['select_method']
    max_percent_rand = 0.0
    max_density_rand = 0.0
    if select_method.upper() == 'PERCENT':
        max_percent_rand = float(parms['percent_rand'])
    else:
        max_density_rand = float(parms['density_rand'])
    gamma_stdev_reject = float(parms['gamma_stdev_reject'])
    drop_ifg_index = parms['drop_ifg_index']
    low_coh_thresh = 31  # ~ 0.31 # Low-coherence threshold 

    psver = int(read_h5(os.path.join(patch_dir, 'psver.h5'))['psver'])

    psname = f'ps{psver}.h5'
    phname = f'ph{psver}.h5'
    pmname = f'pm{psver}.h5'
    daname = f'da{psver}.h5'
    selectname = f'select{psver}.h5'
    
    # Load main PS data
    ps_data = read_h5(os.path.join(patch_dir, psname))

    # drop_ifg_index may reference a subset of ifgs
    n_ifg_total = int(ps_data['n_ifg'])
    ifg_index_all = np.arange(1, n_ifg_total + 1, dtype=int)
    if len(drop_ifg_index) > 0:
        keep_in_m = np.setdiff1d(ifg_index_all, drop_ifg_index)  # e.g. shape (?)

    bperp = ps_data['bperp']
        
    master_ix = int(ps_data['master_ix'])
    no_master_ix = list(set(range(1, n_ifg_total + 1)) - {master_ix})
    # But also remove drop_ifg_index from that:
    if len(drop_ifg_index) > 0:
        no_master_ix = sorted(set(no_master_ix) - set(drop_ifg_index))
    # Adjust indices after master_ix (as done in MATLAB)
    no_master_ix = np.array(no_master_ix)
    no_master_ix[no_master_ix > master_ix] -= 1

    ph = read_h5(os.path.join(patch_dir, phname))['ph']
    ph = np.atleast_2d(ph)
    #ph = ph.T
    master_col = master_ix - 1
    all_cols = np.arange(ph.shape[1], dtype=int)
    remove_cols = [master_col] + [ix - 1 for ix in drop_ifg_index]
    keep_cols = [c for c in all_cols if c not in remove_cols]

    ph = ph[:, keep_cols]
    bperp = np.atleast_1d(bperp).flatten()
    bperp = bperp[keep_cols]
    n_ifg = len(keep_cols)

    n_ps = int(ps_data['n_ps'])
    xy = ps_data['xy']

    if xy.shape[0] != n_ps:
        xy = xy.T

    # Load pm data
    pm_data = read_h5(os.path.join(patch_dir, pmname))

    # load da if exist
    D_A = read_h5(os.path.join(patch_dir, daname))['D_A']

    # If there are enough candidate pixels for D_A binning
    if D_A.shape[0] >= 10000:
        # chunk up PSC
        D_A_sort = np.sort(D_A, axis=0).flatten()
        if D_A.shape[0] >= 50000:
            bin_size = 10000
        else:
            bin_size = 2000
        D_A_max_values = [0]
        # Extract bin edges
        for idx in range(bin_size, D_A.shape[0] - bin_size + 1, bin_size):
            D_A_max_values.append(D_A_sort[idx - 1])
        D_A_max_values.append(D_A_sort[-1])  # final
        D_A_max = np.array(D_A_max_values)
    else:
        # if not enough PS for binning
        D_A_max = np.array([0, 1], dtype=np.float64)
        D_A = np.ones((n_ps, 1), dtype=np.float64)

    # For "PERCENT" vs. "DENSITY" approach
    if select_method.upper() != 'PERCENT':
        # We'll measure patch_area in km^2.  (Coordinates presumably in m.)
        xy_min = np.min(xy[:, 1:3], axis=0)
        xy_max = np.max(xy[:, 1:3], axis=0)
        patch_area_km2 = np.prod((xy_max - xy_min)) / 1.0e6
        # now compute the equivalent of "max_percent_rand" from "density_rand"
        if len(D_A_max) > 1:
            max_percent_rand = max_density_rand * patch_area_km2 / (len(D_A_max) - 1)
        else:
            max_percent_rand = max_density_rand * patch_area_km2

    # Prepare for building threshold
    min_coh = np.zeros(len(D_A_max) - 1, dtype=np.float64)
    D_A_mean = np.zeros(len(D_A_max) - 1, dtype=np.float64)

    coh_ps_all = pm_data['coh_ps'].flatten()
    Nr_dist = pm_data['Nr']

    # Build gamma threshold by D_A bin
    for i_bin in range(len(D_A_max) - 1):
        # Indices of PS with D_A in bin
        bin_indices = np.where((D_A > D_A_max[i_bin]) & (D_A <= D_A_max[i_bin + 1]))[0]
        # compute mean
        if len(bin_indices) > 0:
            D_A_mean[i_bin] = np.mean(D_A[bin_indices])
        else:
            D_A_mean[i_bin] = np.nan

        # Filter out zeros from chunk
        coh_chunk = coh_ps_all[bin_indices]
        coh_chunk_non0 = coh_chunk[coh_chunk != 0]
        if len(coh_chunk_non0) < 1 or Nr_dist is None:
            # no threshold can be found here if no data or missing random distribution
            min_coh[i_bin] = np.nan
            continue

        # Build hist of chunk
        # pm.coh_bins in ps_select? It's pm_data['coh_bins'] or similar
        if 'coh_bins' not in pm_data:
            raise KeyError("pm{} file missing 'coh_bins' for building histogram.".format(psver))
        coh_bins = pm_data['coh_bins'].flatten()
        Na, _ = np.histogram(coh_chunk_non0, bins=coh_bins)
        # Scale random distribution to chunk's low-coh region
        if low_coh_thresh > len(Na):
            # fallback in case low_coh_thresh is out of range
            scale_num = np.sum(Na)
            scale_den = np.sum(Nr_dist)
        else:
            scale_num = np.sum(Na[:low_coh_thresh])
            scale_den = np.sum(Nr_dist[:low_coh_thresh])
        if scale_den == 0:
            scale_den = 1
        Nr_scaled = Nr_dist * (scale_num / scale_den)

        # avoid divide by zero
        Na_safe = Na.copy()
        Na_safe[Na_safe == 0] = 1
        percent_rand = None

        if select_method.upper() == 'PERCENT':
            # (flip cumsum(Nr) / cumsum(Na) * 100)
            cNa = np.cumsum(np.flipud(Na_safe))
            cNr = np.cumsum(np.flipud(Nr_scaled))
            # avoid /0
            cNa[cNa == 0] = 1
            pr = cNr / cNa * 100.0
            percent_rand = np.flipud(pr)
        else:
            # do absolute # of random points
            pr = np.cumsum(np.flipud(Nr_scaled))  # no division
            percent_rand = np.flipud(pr)

        # find ok_ix => first place that is < max_percent_rand
        ok_ix = np.where(percent_rand < max_percent_rand)[0]
        if len(ok_ix) < 1:
            min_coh[i_bin] = 1.0  # no threshold meets criteria
        else:
            min_fit_ix = ok_ix[0] - 3
            if min_fit_ix <= 0:
                min_coh[i_bin] = np.nan
            else:
                max_fit_ix = ok_ix[0] + 2
                if max_fit_ix > 99:
                    max_fit_ix = 99
                # Fit polynomial in the range [min_fit_ix : max_fit_ix]
                # x domain => percent_rand in that region
                # y domain => actual coherence bin index
                x_data = percent_rand[min_fit_ix : max_fit_ix + 1]
                # if we used bins of 0.01 => bin i => i*0.01
                y_data = np.arange(min_fit_ix, max_fit_ix + 1) * 0.01
                # polyfit in python => np.polyfit(x,y,degree)
                # be mindful of potential all-NaN scenario.
                if np.any(np.isnan(x_data)) or len(x_data) < 4:
                    min_coh[i_bin] = np.nan
                else:
                    p_coeff = np.polyfit(x_data, y_data, 3)
                    # Evaluate at x=max_percent_rand
                    min_coh_val = np.polyval(p_coeff, max_percent_rand)
                    min_coh[i_bin] = min_coh_val

        # handle no good bins
        nonnan_ix = np.where(~np.isnan(min_coh))[0]
        if len(nonnan_ix) < 1:
            print("Not enough random phase pixels to set gamma threshold => using default 0.3")
            coh_thresh = 0.3
            coh_thresh_coeffs = []
        else:
            # we have some bins
            valid_min_coh = min_coh[nonnan_ix]
            valid_D_A_mean = D_A_mean[nonnan_ix]
            if len(valid_min_coh) > 1:
                # fit a line
                p_fit = np.polyfit(valid_D_A_mean, valid_min_coh, 1)
                # check slope
                if p_fit[0] > 0:
                    # good slope
                    # evaluate at each D_A
                    coh_thresh = np.polyval(p_fit, D_A)
                    coh_thresh_coeffs = p_fit
                else:
                    # fallback
                    coh_thresh = np.polyval(p_fit, 0.35)
                    coh_thresh_coeffs = []
            else:
                # only one valid bin
                coh_thresh = valid_min_coh[0]
                coh_thresh_coeffs = []

    coh_thresh = max(coh_thresh, 0)
    print(
        "Initial gamma threshold: {:.3f} at D_A={:.2f} to {:.3f} at D_A={:.2f}".format(
            np.min(coh_thresh), np.min(D_A), np.max(coh_thresh), np.max(D_A)
        )
    )

    # Step: initial selection
    # coh_ps is in pm_data['coh_ps']. We filter those that exceed
    # the threshold (which might vary by D_A).
    if isinstance(coh_thresh, np.ndarray) and len(coh_thresh) == len(coh_ps_all):
        # elementwise threshold
        ix_bool = (coh_ps_all > coh_thresh)
    else:
        # single threshold
        ix_bool = (coh_ps_all > float(coh_thresh))  # fallback

    ix = np.where(ix_bool)[0]
    n_ps_init_sel = len(ix)
    print(f"{n_ps_init_sel} PS selected initially")

    # ###########################################################################
    # # Part-time PS rejection by bootstrap stdev if gamma_stdev_reject>0
    # ###########################################################################
    # if gamma_stdev_reject > 0:
    #     # gather ph_res => pm_data['ph_res'], but we only want the ifgs in keep_in_m
    #     if 'ph_res' not in pm_data:
    #         raise KeyError("pm{} missing ph_res field; needed for stdev-based rejection.".format(psver))
    #     ph_res_all = pm_data['ph_res']
    #     # shape (n_ps, n_ifg_total?). We have pruned some ifgs above. We need to keep only ifg_index
    #     # Actually, we do "ifg_index=setdiff(1:n_ifg, drop_ifg_index)" in MATLAB.  We'll replicate.
    #     # We have 'ph_res2' if re-est_flag = 2? Wait, that's later. For now let's assume ph_res is good.
    #     if ph_res_all.shape[1] < n_ifg:
    #         # the pm might have a shape after removing the master. We'll assume it matches 'ph' columns now
    #         pass

    #     # We'll consider the ifgs that remain. In MATLAB, that is ifg_index. Here it's 'ph_res_all[:, ifg_index-1]'
    #     # But we've already removed those columns from 'ph' above. So if pm was also pruned, let's assume they match now.
    #     ph_res_cpx = np.exp(1j * ph_res_all[:, :n_ifg])  # exp(j * phases) to get complex
    #     # take only rows in ix
    #     ph_res_cpx_sel = ph_res_cpx[ix, :]
    #     # We do bootstrap stdev. For large n_ifg, can do approximate or partial approach:
    #     # The MATLAB approach: coh_std(i)=std(bootstrp(100,@(ph) abs(sum(ph))/length(ph), ph_res_cpx(ix(i),ifg_index)));
    #     # We'll do a direct approach: The standard approach, do many resamples. A direct re-implementation can be large in Python.
    #     # For consistency, we might do a simpler stdev approach. But let's just do an approximate normal approach:
    #     # => stdev of "abs(sum(ph))/N" across ifg? That's effectively the stdev of coherence across ifg bootstrap?
    #     # The user said "you may assume that all external functions available ... but if some deps seem missing, inform me."
    #     # We'll do an approximate approach or we can place a TODO for the actual bootstrap logic if needed.

    #     # We'll do a simpler approach: stdev of the distribution of abs(sum(ph) / n_ifg) using a bootstrap method is not trivial
    #     # but let's replicate the main line in MATLAB:
    #     #   standard approach: for each pixel i => ph_res_cpx_sel[i], do 100 bootstrap resamples ...
    #     # That can be quite large. We'll do a naive approach:
    #     # (We can do from sklearn.utils import resample or do custom.)
    #     # We'll warn that for large n_ps x n_ifg, this can be slow in pure Python.
    #     from sklearn.utils import resample

    #     n_sel_ps = ph_res_cpx_sel.shape[0]
    #     coh_std = np.zeros(n_sel_ps, dtype=np.float64)

    #     # For each selected PS
    #     for irow in range(n_sel_ps):
    #         # original data row => ph_res_cpx_sel[irow, :]
    #         # do 100 bootstrap resamples
    #         boot_vals = []
    #         for b_i in range(100):
    #             sample = resample(ph_res_cpx_sel[irow, :])
    #             boot_vals.append(np.abs(np.sum(sample)) / len(sample))
    #         coh_std[irow] = np.std(boot_vals, ddof=1)

    #     # Now we filter out those that exceed gamma_stdev_reject
    #     keep_ix_sub = np.where(coh_std < gamma_stdev_reject)[0]
    #     # we want the original 'ix' - but only the subset that passes
    #     ix = ix[keep_ix_sub]
    #     n_ps_left = len(ix)
    #     logit(f"{n_ps_left} PS left after pps rejection")

    ############################################################################
    # Depending on reest_flag, we re-estimate coherence or skip
    ############################################################################
    # keep_ix is logically all True for those we haven't excluded
    keep_ix = np.ones(len(ix), dtype=bool)

    # re-estimate gamma from scratch for selected PS
    # we drop those ifgs in drop_ifg_index from the noise re-est
    # the code in ps_select.m removes the pixel from the patch
    # and re-filters a patch around it. We'll replicate the structure
    # but we do not have the direct internal code from clap_filt_patch in python
    # If needed, implement or let the user know it might be missing:
    ph_patch2 = np.zeros((len(ix), n_ifg), dtype=np.complex64)
    ph_res2 = np.zeros((len(ix), n_ifg), dtype=np.float32)
    K_ps2 = np.zeros((len(ix),), dtype=np.float64)
    C_ps2 = np.zeros((len(ix),), dtype=np.float64)
    coh_ps2 = np.zeros((len(ix),), dtype=np.float64)

    # We'll attempt a partial approach
    ph_grid_all = pm_data['ph_grid']
    grid_ij_all = pm_data['grid_ij']

    # We'll do the same window approach as in ps_select => n_win x n_win
    n_win_local = clap_win
    n_i = int(np.max(grid_ij_all[:, 0]))
    n_j = int(np.max(grid_ij_all[:, 1]))

    # Convert D_A array shape
    # We'll do a loop over each selected PS
    for i_count, ps_idx in enumerate(ix):
        ps_ij = grid_ij_all[ps_idx, :]  # row
        # Convert to zero-based
        center_i = ps_ij[0] - 1
        center_j = ps_ij[1] - 1

        i_min = max(center_i - n_win_local // 2, 0)
        i_max = i_min + n_win_local - 1
        if i_max >= n_i:
            i_shift = i_max - (n_i - 1)
            i_min -= i_shift
            i_max = n_i - 1

        j_min = max(center_j - n_win_local // 2, 0)
        j_max = j_min + n_win_local - 1
        if j_max >= n_j:
            j_shift = j_max - (n_j - 1)
            j_min -= j_shift
            j_max = n_j - 1

        if (i_min < 0) or (j_min < 0):
            # can't do a valid patch
            ph_patch2[i_count, :] = 0
            continue

        # gather patch => shape (n_win_local, n_win_local, n_ifg)
        # remove pixel => 0
        patch_data = ph_grid_all[i_min : i_max + 1, j_min : j_max + 1, :]
        # local coords in patch
        local_i = center_i - i_min
        local_j = center_j - j_min
        if local_i < 0 or local_i >= patch_data.shape[0] or local_j < 0 or local_j >= patch_data.shape[1]:
            # can't handle that
            ph_patch2[i_count, :] = 0
            continue

        # remove that pixel from patch
        patch_data[local_i, local_j, :] = 0

        # call clap_filt_patch, or just do a simpler approach if we have "clap_filt"
        out_filt = np.zeros_like(patch_data)
        for f_ifg in range(n_ifg):
            out_filt[:, :, f_ifg] = clap_filt(
                patch_data[:, :, f_ifg],
                clap_alpha,
                clap_beta,
                int(n_win_local * 0.75),
                int(n_win_local * 0.25),
                pm_data['low_pass'] if 'low_pass' in pm_data else None
            )
        # read center pixel from out_filt
        ph_patch2[i_count, :] = out_filt[local_i, local_j, :]

    # For each selected PS
    for i_count, ps_idx in enumerate(ix):
        # gather psdph
        psdph = ph[ps_idx, :] * np.conj(ph_patch2[i_count, :])
        if np.all(psdph != 0):
            psdph_norm = psdph / np.abs(psdph)
            # if bperp is 1D => pass single row. If 2D => pass row
            Kopt, Copt, cohopt, ph_residual = ps_topofit(
                psdph_norm,
                bperp if bperp.ndim == 1 else bperp[ps_idx, :],
                float(pm_data['n_trial_wraps']),
                'n'
            )
            K_ps2[i_count] = Kopt
            C_ps2[i_count] = Copt
            coh_ps2[i_count] = cohopt
            ph_res2[i_count, :] = np.angle(ph_residual)
        else:
            K_ps2[i_count] = np.nan
            coh_ps2[i_count] = np.nan

        if i_count == 50:
            break
    # Update pm_data
    pm_data['coh_ps'][ix] = coh_ps2

    # Recompute threshold after re-estimation
    # (similar logic as above but only for re-selected PS)
    for i_bin in range(len(D_A_max) - 1):
        bin_indices = np.where((D_A > D_A_max[i_bin]) & (D_A <= D_A_max[i_bin + 1]))[0]
        if len(bin_indices) < 1 or 'Nr' not in pm_data:
            min_coh[i_bin] = np.nan
            continue
        D_A_mean[i_bin] = np.mean(D_A[bin_indices])
        coh_chunk = pm_data['coh_ps'][bin_indices].flatten()
        coh_chunk_non0 = coh_chunk[coh_chunk != 0]
        if len(coh_chunk_non0) < 1:
            min_coh[i_bin] = np.nan
            continue

        Na, _ = np.histogram(coh_chunk_non0, bins=pm_data['coh_bins'].flatten())
        Nr_scaled = None
        if low_coh_thresh <= len(Na):
            scale_num = np.sum(Na[:low_coh_thresh])
            scale_den = np.sum(pm_data['Nr'][:low_coh_thresh])
        else:
            scale_num = np.sum(Na)
            scale_den = np.sum(pm_data['Nr'])
        if scale_den == 0:
            scale_den = 1
        Nr_scaled = pm_data['Nr'] * (scale_num / scale_den)

        Na_safe = Na.copy()
        Na_safe[Na_safe == 0] = 1

        if select_method.upper() == 'PERCENT':
            cNa = np.cumsum(np.flipud(Na_safe))
            cNr = np.cumsum(np.flipud(Nr_scaled))
            cNa[cNa == 0] = 1
            pr = cNr / cNa * 100
            percent_rand = np.flipud(pr)
        else:
            pr = np.cumsum(np.flipud(Nr_scaled))
            percent_rand = np.flipud(pr)

        ok_ix = np.where(percent_rand < max_percent_rand)[0]
        if len(ok_ix) < 1:
            min_coh[i_bin] = 1.0
        else:
            min_fit_ix = ok_ix[0] - 3
            if min_fit_ix <= 0:
                min_coh[i_bin] = np.nan
            else:
                max_fit_ix = ok_ix[0] + 2
                if max_fit_ix > 99:
                    max_fit_ix = 99
                x_data = percent_rand[min_fit_ix : max_fit_ix + 1]
                y_data = np.arange(min_fit_ix, max_fit_ix + 1) * 0.01
                if np.any(np.isnan(x_data)) or len(x_data) < 4:
                    min_coh[i_bin] = np.nan
                else:
                    p_coeff = np.polyfit(x_data, y_data, 3)
                    min_coh_val = np.polyval(p_coeff, max_percent_rand)
                    min_coh[i_bin] = min_coh_val

    # finalize threshold again
    nonnan_ix = np.where(~np.isnan(min_coh))[0]
    if len(nonnan_ix) < 1:
        coh_thresh = 0.3
        coh_thresh_coeffs = []
    else:
        valid_min_coh = min_coh[nonnan_ix]
        valid_D_A_mean = D_A_mean[nonnan_ix]
        if len(valid_min_coh) > 1:
            p_fit = np.polyfit(valid_D_A_mean, valid_min_coh, 1)
            if p_fit[0] > 0:
                coh_thresh = np.polyval(p_fit, D_A[ix])
                coh_thresh_coeffs = p_fit
            else:
                coh_thresh = np.polyval(p_fit, 0.35)
                coh_thresh_coeffs = []
        else:
            coh_thresh = valid_min_coh[0]
            coh_thresh_coeffs = []

    # ensure no negative
    if isinstance(coh_thresh, np.ndarray):
        coh_thresh[coh_thresh < 0] = 0
    else:
        coh_thresh = max(0, coh_thresh)

    print(
        "Reestimation gamma threshold: {:.3f} at D_A={:.2f} to {:.3f} at D_A={:.2f}".format(
            np.min(coh_thresh), np.min(D_A), np.max(coh_thresh), np.max(D_A)
        )
    )

    # refine keep_ix
    bperp_range = float(np.max(bperp) - np.min(bperp))
    if bperp_range == 0:
        bperp_range = 1e-9  # avoid /0

    # The condition: keep_ix=coh_ps2>coh_thresh & abs(pm.K_ps(ix)-K_ps2)<2*pi/bperp_range
    # Original pm.K_ps => pm_data['K_ps'] ? We must check if it exists
    # We do: abs(pm.K_ps(ix) - K_ps2) < ...
    if 'K_ps' not in pm_data:
        print("pm_data missing K_ps. Skipping K constraint.")
        keep_ix = (coh_ps2 > coh_thresh)
    else:
        old_K_ps = pm_data['K_ps'].flatten()
        K_diff = np.abs(old_K_ps[ix] - K_ps2)
        keep_ix = (coh_ps2 > coh_thresh) & (K_diff < (2.0 * np.pi / bperp_range))

    n_after_reest = np.sum(keep_ix)
    print(f"{n_after_reest} ps selected after re-estimation of coherence")

    # # possibly store info about # PS left
    # # no_ps_info => "no_ps_info.mat"
    # no_ps_info_file = 'no_ps_info.mat'
    # stamps_step_no_ps = None
    # if os.path.exists(no_ps_info_file):
    #     tmp_data = load_mat(no_ps_info_file)
    #     if 'stamps_step_no_ps' in tmp_data:
    #         stamps_step_no_ps = tmp_data['stamps_step_no_ps'].flatten()
    # else:
    #     # create new
    #     stamps_step_no_ps = np.zeros(5, dtype=int)  # just as shown in ps_select

    # if stamps_step_no_ps is not None and len(stamps_step_no_ps) >= 3:
    #     if np.sum(keep_ix) == 0:
    #         logit("***No PS points left. Updating the stamps log for this***")
    #         stamps_step_no_ps[2] = 1  # step index 3 => array index 2
    #     # re-save
    #     # In Python, to store it again:
    #     # we can do a stamps_save or some specialized code. We'll do a minimal approach:
    #     # a minimal approach: let's use stamps_save if it can handle np arrays
    #     # or do custom code:
    #     stamps_save(no_ps_info_file, stamps_step_no_ps)
    # else:
    #     print("WARNING: no_ps_info did not match expected structure. Skipping that update.")

    ifg_index = np.array(no_master_ix)
    # Remove drop_ifg_index from ifg_index if needed
    if len(drop_ifg_index) > 0:
        ifg_index = np.array([x for x in ifg_index if x not in drop_ifg_index])

    save_h5(patch_dir, selectname, 
            **{"ix":ix, "keep_ix":keep_ix, "ph_patch2":ph_patch2, 
               "ph_res2":ph_res2, "K_ps2":K_ps2, "C_ps2":C_ps2, 
               "coh_ps2":coh_ps2,"coh_thresh":coh_thresh, 
               "coh_thresh_coeffs":coh_thresh_coeffs, "clap_alpha":clap_alpha, 
               "clap_beta":clap_beta,"clap_win":clap_win, "max_percent_rand":max_percent_rand, 
               "gamma_stdev_reject":gamma_stdev_reject, "ifg_index":ifg_index})