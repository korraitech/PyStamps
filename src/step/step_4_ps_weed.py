import numpy as np
import os
from utils import read_h5,save_h5


def step_4_ps_weed(workdir:str, patch:str) -> None:

    print("Running Step-4 ...\t[{}]".format(patch))
    print('Weeding selected pixels...')

    psver = read_h5(os.path.join(workdir, patch, 'psver.h5'))['psver'][0][0]
    psname = f'ps{psver}.h5'
    pmname = f'pm{psver}.h5'
    phname = f'ph{psver}.h5'
    selectname = f'select{psver}.h5'
    hgtname = f'hgt{psver}.h5'
    laname = f'la{psver}.h5'
    incname = f'inc{psver}.h5'
    bpname = f'bp{psver}.h5'

    ps = read_h5(os.path.join(workdir, patch, psname))
    drop_ifg_index = []
    ifg_index = np.setdiff1d(np.arange(1, ps['n_ifg'][0][0] + 1), drop_ifg_index)

    if os.path.exists(os.path.join(workdir, patch, phname)):
        ph_data = read_h5(os.path.join(workdir, patch, phname))
        ph = ph_data['ph']
    else:
        ph = ps['ph']

    sl = read_h5(os.path.join(workdir, patch, selectname))
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

    pm = read_h5(os.path.join(workdir, patch, pmname))
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

    if os.path.exists(os.path.join(workdir, patch, hgtname)):
        hgt = read_h5(os.path.join(workdir, patch, hgtname))
        hgt = hgt['hgt'][ix2]
    
    n_ps_other = 0
    n_ps_low_D_A = len(ix2)
    n_ps = n_ps_low_D_A + n_ps_other
    ix_weed = np.ones(n_ps, dtype=bool)
    print(f"{n_ps_low_D_A} low D_A PS, {n_ps_other} high D_A PS")

    xy_weed = xy2[ix_weed,:]
    # update PS inofmration
    n_ps = sum(ix_weed)

    # Remove dupplicated points
    # Some non-adjacent pixels are allocated the same lon/lat by DORIS.
    # If duplicates occur, the pixel with the highest coherence is kept.
    ix_weed_num = np.where(ix_weed)[0]
    _, I = np.unique(xy_weed[:,2:3], axis=0)
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
    n_ps = np.sum(ix_weed)
    ix_weed2 = np.ones(n_ps, dtype=bool)
    
    # Weedign noisy pixels
    ps_std = np.zeros(n_ps)
    ps_max = np.zeros(n_ps)

    # Save the results
    weedname = f'weed_rahul_{psver}.h5'
    save_h5(weedname, **{'ix_weed': ix_weed, 'ix_weed2': ix_weed2, 
                       'ps_std': ps_std, 'ps_max': ps_max, 'ifg_index': ifg_index})
    
    coh_ps = coh_ps2[ix_weed]
    K_ps = K_ps2[ix_weed]
    C_ps = C_ps2[ix_weed]
    ph_patch = ph_patch2[ix_weed,:]
    if ph_res2 is not None:
        ph_res = ph_res2[ix_weed,:]
    else:
        ph_res = ph_res2
    
    pmname = f'pm_rahul_{psver}.h5'
    save_h5(pmname, **{'ph_patch': ph_patch, 'ph_res': ph_res, 
                     'coh_ps': coh_ps, 'K_ps': K_ps, 'C_ps': C_ps})
    
    phname = f'ph_rahul_{psver}.h5'
    ph2 = ph2[ix_weed,:]
    save_h5(phname, **{'ph': ph2})

    xy2 = xy2[ix_weed,:]
    ij2 = ij2[ix_weed,:]
    lonlat2 = lonlat2[ix_weed,:]
    ps['xy'] = xy2
    ps['ij'] = ij2
    ps['lonlat'] = lonlat2
    ps['n_ps'] = n_ps
    psname = f'ps_rahul_{psver}.h5'
    save_h5(psname, **ps)

    if os.path.exists(os.path.join(workdir, patch, hgtname)):
        hgt = read_h5(os.path.join(workdir, patch, hgtname))
        hgt = hgt['hgt'][ix_weed]
        save_h5(f'hgt_rahul_{psver}.h5', **{'hgt': hgt})
    
    if os.path.exists(os.path.join(workdir, patch, laname)):
        la = read_h5(os.path.join(workdir, patch, laname))
        la = la['la'][ix2]
        la = la[ix_weed]
        save_h5(f'la_rahul_{psver}.h5', **{'la': la})
        
    if os.path.exists(os.path.join(workdir, patch, incname)):
        inc = read_h5(os.path.join(workdir, patch, incname))
        inc = inc['inc'][ix2]
        inc = inc[ix_weed]
        save_h5(f'inc_rahul_{psver}.h5', **{'inc': inc})
        
    if os.path.exists(os.path.join(workdir, patch, bpname)):
        bp = read_h5(os.path.join(workdir, patch, bpname))
        bperp_mat = bp['bperp_mat'][ix2,:]
        bperp_mat = bperp_mat[ix_weed,:]
        save_h5(f'bp_rahul_{psver}.h5', **{'bperp_mat': bperp_mat})
    
    save_h5(os.path.join(workdir, patch, f'psver_rahul_{psver}.h5'), **{'psver': psver+1})

workdir = '/home/ubuntu/workspace/enhancement/enh_stamps/export_rslc_small/small_pystamps'
patch = 'PATCH_1'
step_4_ps_weed(workdir, patch)
