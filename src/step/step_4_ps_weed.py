import numpy as np
import os
from scipy import spatial
from .utils import read_h5,save_h5

def lscov(G, y, w):
    sqrt_w = np.sqrt(w)[:, np.newaxis]
    return np.linalg.lstsq(G * sqrt_w, y * sqrt_w)[0]

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
        edges_all = np.vstack(( tri_simplices[:, [0, 1]],
                                tri_simplices[:, [1, 2]],
                                tri_simplices[:, [2, 0]]))
        edges_sorted = np.sort(edges_all, axis=1)
        edgs = np.unique(edges_sorted, axis=0)
        n_edge = edgs.shape[0]
        
        ph_weed = ph2[ix_weed, :] * np.exp(-1j * (np.outer(K_ps2[ix_weed], bperp.T)))
        ph_weed = ph_weed / np.abs(ph_weed)
        ph_weed[:, int(ps['master_ix'])] = np.exp(1j * (C_ps2[ix_weed]))
        dph_space = ph_weed[edgs[:, 1], :] * np.conj(ph_weed[edgs[:, 0], :])
        dph_space = dph_space[:, ifg_index]
        n_use = len(ifg_index)

        print('Estimating noise for all arcs...')
        dph_smooth1 = np.zeros((n_edge, n_use), dtype=np.complex64)
        dph_smooth2 = np.zeros((n_edge, n_use), dtype=np.complex64)

        for i1 in range(n_use):
            time_diff = day[ifg_index[i1]] - day[ifg_index]
            weight_factor = np.exp(-(time_diff**2) / (2 * weed_time_win**2))
            weight_factor = weight_factor / weight_factor.sum()

            dph_mean = np.sum(dph_space * np.tile(weight_factor, (n_edge, 1)), axis=1)
            dph_mean_adj = np.angle(dph_space * np.tile(np.conj(dph_mean)[:, np.newaxis], (1, n_use)))

            G = np.column_stack((np.ones(n_use), time_diff))
            m1 = lscov(G,dph_mean_adj.T,weight_factor)

            dph_mean_adj = np.angle(np.exp(1j * (dph_mean_adj - (G @ m1).T)))
            m2 = lscov(G,dph_mean_adj.T,weight_factor)

            weight_factor[i1] = 0
            dph_smooth1[:,i1] = dph_mean * np.exp(1j * (m1[0,:] + m2[0,:]))
            dph_smooth2[:,i1] = np.sum(dph_space * weight_factor.reshape(1, -1), axis=1)

        dph_noise1 = np.angle(dph_space * np.conj(dph_smooth1))
        dph_noise2 = np.angle(dph_space * np.conj(dph_smooth2))
        ifg_var = np.var(dph_noise2, axis=0, ddof=0)

        K = lscov(bperp[ifg_index][:, np.newaxis], dph_noise1.T, 1.0 / ifg_var).T
        dph_noise1 = dph_noise1 - K @ (bperp[ifg_index][:, np.newaxis]).T

        edge_std = np.zeros((n_edge, 1))
        edge_max = np.zeros((n_edge, 1))
        edge_std = np.std(dph_noise1, axis=1, keepdims=True)
        edge_max = np.max(np.abs(dph_noise1), axis=1, keepdims=True)

        print('Estimating max noise for all pixels...')
        ps_std = np.full(n_ps, np.inf, dtype=np.float64)
        ps_max = np.full(n_ps, np.inf, dtype=np.float64)
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
        hgt = hgt['hgt'][ix2]
        hgt = hgt[ix_weed]
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
