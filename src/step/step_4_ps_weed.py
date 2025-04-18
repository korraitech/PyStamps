import numpy as np
import os
from scipy import spatial
from .utils import read_h5,save_h5

def step_4_ps_weed(workdir:str, patch:str, parms:dict) -> None:

    print("Running Step-4 ...\t[{}]".format(patch))
    print('Weeding selected pixels...')
    patch_dir = os.path.join(workdir,patch)

    weed_time_win = int(parms['weed_time_win'])
    weed_standard_dev = float(parms['weed_standard_dev'])
    weed_max_noise = float(parms['weed_max_noise'])

    psver = int(read_h5(os.path.join(patch_dir, 'psver.h5'))['psver'])
    psname = f'ps{psver}.h5'
    pmname = f'pm{psver}.h5'
    phname = f'ph{psver}.h5'
    selectname = f'select{psver}.h5'
    hgtname = f'hgt{psver}.h5'
    laname = f'la{psver}.h5'
    incname = f'inc{psver}.h5'
    bpname = f'bp{psver}.h5'

    ps = read_h5(os.path.join(patch_dir, psname))
    ifg_index = np.arange(0, int(ps['n_ifg']))

    if os.path.exists(os.path.join(patch_dir, phname)):
        ph_data = read_h5(os.path.join(patch_dir, phname))
        ph = ph_data['ph']
    else:
        ph = ps['ph']

    sl = read_h5(os.path.join(patch_dir, selectname))
    if 'keep_ix' in sl:
        ix2 = sl['ix'][sl['keep_ix']]
        K_ps2 = sl['K_ps2'][sl['keep_ix']]
        C_ps2 = sl['C_ps2'][sl['keep_ix']]
        coh_ps2 = sl['coh_ps2'][sl['keep_ix']]
    else:
        ix2 = sl['ix2']
        K_ps2 = sl['K_ps2']
        C_ps2 = sl['C_ps2']
        coh_ps2 = sl['coh_ps2']
    
    ix2 = ix2.squeeze().astype(int)
    ij2 = ps['ij'][ix2,:]
    xy2 = ps['xy'][ix2,:]
    ph2 = ph[ix2,:]
    lonlat2 = ps['lonlat'][ix2,:]

    pm = read_h5(os.path.join(patch_dir, pmname))
    ph_patch2 = pm['ph_patch'][ix2,:]
    if 'ph_res2' in sl:
        ph_res2 = sl['ph_res2'][sl['keep_ix'],:]
    else:
        ph_res2 = []

    if 'ph' in ps:
        ps.pop('ph', None)

    keys_to_remove = ['xy', 'ij', 'lonlat', 'sort_ix']
    for key in keys_to_remove:
        ps.pop(key, None)

    if os.path.exists(os.path.join(patch_dir, hgtname)):
        hgt = read_h5(os.path.join(patch_dir, hgtname))
        hgt = hgt['hgt'][ix2]

    n_ps_low_D_A = len(ix2)
    n_ps = n_ps_low_D_A
    ix_weed = np.ones(n_ps, dtype=bool)
    print(f"{n_ps_low_D_A} low D_A PS")

    xy_weed = xy2[ix_weed,:]
    # update PS inofmration
    n_ps = sum(ix_weed)

    # Remove dupplicated points
    # Some non-adjacent pixels are allocated the same lon/lat by DORIS.
    # If duplicates occur, the pixel with the highest coherence is kept.
    ix_weed_num = np.where(ix_weed)[0]
    _, I = np.unique(xy_weed[:, 1:3], axis=0, return_index=True)
    dups = np.setdiff1d(I, ix_weed_num)

    for i in dups:
        dups_ix_weed = np.where(
            (xy_weed[:, 1] == xy_weed[i, 1]) & 
            (xy_weed[:, 2] == xy_weed[i, 2])
        )[0]
        dups_ix = ix_weed[dups_ix_weed]
        max_coh_index = np.argmax(coh_ps2[dups_ix])
        remove_mask = np.ones(len(dups_ix_weed), dtype=bool)
        remove_mask[max_coh_index] = False
        ix_weed[dups_ix_weed[remove_mask]] = 0

    if dups is not None and len(dups) > 0:
        xy_weed = xy2[ix_weed, :]
        print(f"{len(dups)} PS with duplicate lon/lat dropped")
    
    # update PS inofmration
    n_ps = int(np.sum(ix_weed))
    ix_weed2 = np.ones(n_ps, dtype=bool)
    
    # Weedign noisy pixels
    ps_std = np.zeros(n_ps)
    ps_max = np.zeros(n_ps)
    day = ps['day']
    bperp = ps['bperp']
    epsilon = 1e-12

    # Assuming all variables are already defined before this code block
    if n_ps != 0:
        points = xy_weed[:, 1:3]
        delaunay_result = spatial.Delaunay(points)
        tri_simplices = delaunay_result.simplices
        edges_all = np.vstack((tri_simplices[:, [0, 1]],
                            tri_simplices[:, [1, 2]],
                            tri_simplices[:, [2, 0]]))
        edges_sorted = np.sort(edges_all, axis=1)
        edgs = np.unique(edges_sorted, axis=0)
        n_edge = edgs.shape[0]
        ph2_selected = ph2[ix_weed, :]
        K_ps2_selected = K_ps2[ix_weed]
        range_error_phase = -1j * np.outer(K_ps2_selected, bperp)
        ph_weed = ph2_selected * np.exp(range_error_phase)
        magnitude = np.abs(ph_weed)
        ph_weed = ph_weed / (magnitude + epsilon)
        master_col_idx_py = ps['master_ix'] - 1
        master_noise_selected = C_ps2[ix_weed]
        ph_weed[:, master_col_idx_py] = np.exp(1j * master_noise_selected)
        edge_std = np.zeros(n_edge)
        edge_max = np.zeros(n_edge)
        phase_node2 = ph_weed[edgs[:, 1], :]
        phase_node1 = ph_weed[edgs[:, 0], :]
        dph_space = phase_node2 * np.conj(phase_node1)
        dph_space = dph_space[:, ifg_index]
        n_use = dph_space.shape[1]

        print('Estimating noise for all arcs...')
        dph_smooth = np.zeros((n_edge, n_use), dtype=np.float32)
        dph_smooth2 = np.zeros((n_edge, n_use), dtype=np.float32)

        for i1 in range(n_use):
            time_diff = day[ifg_index[0]] - day[ifg_index]
            weight_factor = np.exp(-(time_diff**2) / (2 * weed_time_win**2))
            sum_weights = np.sum(weight_factor)
            if sum_weights > epsilon:
                weight_factor = weight_factor / sum_weights
            else:
                weight_factor = np.ones(n_use) / n_use
            dph_mean = np.sum(dph_space * weight_factor, axis=1)
            phase_diff_from_mean = dph_space * np.conj(dph_mean[:, np.newaxis])
            dph_mean_adj = np.angle(phase_diff_from_mean)
            G = np.column_stack((np.ones(n_use), time_diff))
            sqrt_W = np.sqrt(np.maximum(weight_factor, epsilon))
            G_w = G * sqrt_W[:, np.newaxis]
            B_w = dph_mean_adj.T.astype(np.float64) * sqrt_W[:, np.newaxis]
            m, residuals, rank, s = np.linalg.lstsq(G_w, B_w, rcond=None)
            trend_estimate = (G @ m).T
            dph_mean_adj = np.angle(np.exp(1j * (dph_mean_adj - trend_estimate)))
            B_w2 = dph_mean_adj.T.astype(np.float64) * sqrt_W[:, np.newaxis]
            m2, residuals2, rank2, s2 = np.linalg.lstsq(G_w, B_w2, rcond=None)
            intercept_correction = m[0, :] + m2[0, :]
            dph_smooth[:, i1] = dph_mean * np.exp(1j * intercept_correction)
            weight_factor_loo = weight_factor.copy()
            weight_factor_loo[i1] = 0
            sum_weights_loo = np.sum(weight_factor_loo)
            if sum_weights_loo > epsilon:
                weight_factor_loo = weight_factor_loo / sum_weights_loo
            else:
                weight_factor_loo[:] = 0
            dph_smooth2[:, i1] = np.sum(dph_space * weight_factor_loo, axis=1)


        dph_noise = np.angle(dph_space * np.conj(dph_smooth))
        dph_noise2 = np.angle(dph_space * np.conj(dph_smooth2))
        ifg_var = np.var(dph_noise2, axis=0, ddof=1)
        A_py = bperp[ifg_index].reshape(-1, 1).astype(np.float64)
        B_py = dph_noise.T.astype(np.float64)
        weights_lscov = (1.0 / ifg_var).astype(np.float64)
        weights_lscov[~np.isfinite(weights_lscov)] = epsilon
        weights_lscov = np.maximum(weights_lscov, epsilon)
        sqrt_W = np.sqrt(weights_lscov)
        A_w = A_py * sqrt_W[:, np.newaxis]
        B_w = B_py * sqrt_W[:, np.newaxis]
        K_solution, _, _, _ = np.linalg.lstsq(A_w, B_w, rcond=None)
        K = K_solution.T
        baselines_row = bperp[ifg_index].reshape(1, -1)
        baseline_error_term = K @ baselines_row
        dph_noise = dph_noise - baseline_error_term
        edge_std = np.std(dph_noise, axis=1, ddof=1)
        edge_max = np.max(np.abs(dph_noise), axis=1)

        print('Estimating max noise for all pixels...')
        ps_std = np.full(n_ps, np.inf, dtype=np.float32)
        ps_max = np.full(n_ps, np.inf, dtype=np.float32)
        for i in range(n_edge):
            point_indices = edgs[i]
            current_edge_std = edge_std[i]
            ps_std[point_indices] = np.minimum(ps_std[point_indices], current_edge_std)
            current_edge_max = edge_max[i]
            ps_max[point_indices] = np.minimum(ps_max[point_indices], current_edge_max)

        ix_weed2 = (ps_std < weed_standard_dev) & (ps_max < weed_max_noise)
        ix_weed[ix_weed] = ix_weed2
        n_ps = np.sum(ix_weed)
        print(f"{n_ps} PS kept after dropping noisy pixels")

    if n_ps == 0:
        print('***No PS points left. Updating the stamps log for this****')

    # Save the results
    weedname = f'weed{psver}.h5'
    save_h5(patch_dir,weedname, **{'ix_weed': ix_weed, 'ix_weed2': ix_weed2, 
                       'ps_std': ps_std, 'ps_max': ps_max, 'ifg_index': ifg_index})
    
    coh_ps = coh_ps2[ix_weed]
    K_ps = K_ps2[ix_weed]
    C_ps = C_ps2[ix_weed]
    ph_patch = ph_patch2[ix_weed,:]
    if ph_res2 is not None:
        ph_res = ph_res2[ix_weed,:]
    else:
        ph_res = ph_res2
    
    psver = psver + 1
    pmname = f'pm{psver}.h5'
    save_h5(patch_dir,pmname, **{'ph_patch': ph_patch, 'ph_res': ph_res, 
                     'coh_ps': coh_ps, 'K_ps': K_ps, 'C_ps': C_ps})
    
    phname = f'ph{psver}.h5'
    ph2 = ph2[ix_weed,:]
    save_h5(patch_dir,phname, **{'ph': ph2})

    xy2 = xy2[ix_weed,:]
    ij2 = ij2[ix_weed,:]
    lonlat2 = lonlat2[ix_weed,:]
    ps['xy'] = xy2
    ps['ij'] = ij2
    ps['lonlat'] = lonlat2
    ps['n_ps'] = n_ps
    psname = f'ps{psver}.h5'
    save_h5(patch_dir,psname, **ps)

    if os.path.exists(os.path.join(patch_dir, hgtname)):
        hgt = read_h5(os.path.join(patch_dir, hgtname))
        hgt = hgt['hgt'][ix_weed]
        save_h5(patch_dir,f'hgt{psver}.h5', **{'hgt': hgt})
    
    if os.path.exists(os.path.join(patch_dir, laname)):
        la = read_h5(os.path.join(patch_dir, laname))
        la = la['la'][ix2]
        la = la[ix_weed]
        save_h5(patch_dir,f'la{psver}.h5', **{'la': la})
        
    if os.path.exists(os.path.join(patch_dir, incname)):
        inc = read_h5(os.path.join(patch_dir, incname))
        inc = inc['inc'][ix2]
        inc = inc[ix_weed]
        save_h5(patch_dir,f'inc{psver}.h5', **{'inc': inc})
        
    if os.path.exists(os.path.join(patch_dir, bpname)):
        bp = read_h5(os.path.join(patch_dir, bpname))
        bperp_mat = bp['bperp_mat'][ix2,:]
        bperp_mat = bperp_mat[ix_weed,:]
        save_h5(patch_dir,f'bp{psver}.h5', **{'bperp_mat': bperp_mat})
    
    save_h5(patch_dir,f'psver.h5', **{'psver': psver})
