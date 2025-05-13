import numpy as np
import datetime
import os
from .utils import read_h5,save_h5
from .llh2local import llh2local

def correct_phase(patch_dir:str,psver:int):
    psname = f'ps{psver}.h5'
    phname = f'ph{psver}.h5'
    pmname = f'pm{psver}.h5'
    bpname = f'bp{psver}.h5'

    pm = read_h5(os.path.join(patch_dir, pmname))
    K_ps = pm['K_ps']
    C_ps = pm['C_ps']

    ps = read_h5(os.path.join(patch_dir, psname))
    master_ix = int(np.sum(ps['master_day'] > ps['day']))
    n_ps = ps['n_ps']

    bp = read_h5(os.path.join(patch_dir, bpname))
    bperp_mat = np.hstack((
        bp['bperp_mat'][:, :master_ix],
        np.zeros((n_ps, 1), dtype=np.float32),
        bp['bperp_mat'][:, master_ix:])
    )
    
    n_ifg = ps['n_ifg']
    ph = read_h5(os.path.join(patch_dir, phname))['ph']

    ph_rc = ph * np.exp(-1j * (np.tile(K_ps.reshape(-1, 1), (1, n_ifg)) * bperp_mat + np.tile(C_ps.reshape(-1, 1), (1, n_ifg))))

    ph_reref = np.hstack((
        pm['ph_patch'][:, :master_ix],
        np.ones((n_ps, 1), dtype=np.complex64),
        pm['ph_patch'][:, master_ix:]
    ))

    rcname = f'rc{psver}.h5'
    save_h5(patch_dir, rcname, **{'ph_rc': ph_rc, 'ph_reref': ph_reref})

def intersect_rows(a, b):
    a_view = a.view([('', a.dtype)] * a.shape[1])
    b_view = b.view([('', b.dtype)] * b.shape[1])
    _, IA, IB = np.intersect1d(a_view, b_view, return_indices=True)
    return IA, IB

def estimate_noise_stddev(workdir:str,psver:int):
    psname = f'ps{psver}.h5'
    phname = f'ph{psver}.h5'
    pmname = f'pm{psver}.h5'
    bpname = f'bp{psver}.h5'
    ifgstdname = f'ifgstd{psver}.h5'

    ps = read_h5(os.path.join(workdir, psname))
    pm = read_h5(os.path.join(workdir, pmname))
    bp = read_h5(os.path.join(workdir, bpname))

    if os.path.exists(os.path.join(workdir,phname)):
        phin = read_h5(os.path.join(workdir,phname))
        ph = phin['ph']
    else:
        ph = ps['ph']
    
    n_ps = len(ps['xy'])
    n_ifg = int(ps['n_ifg'])
    master_ix = int(ps['master_ix'])

    bperp_mat = np.concatenate([
        bp['bperp_mat'][:, :master_ix],
        np.zeros((n_ps, 1), dtype=np.float32),
        bp['bperp_mat'][:, master_ix:]
    ], axis=1)

    ph_patch = np.concatenate([
        pm['ph_patch'][:, 0:master_ix],
        np.ones((n_ps, 1)),
        pm['ph_patch'][:, master_ix:]
    ], axis=1)

    ph_diff = np.angle(ph * np.conj(ph_patch) * np.exp(-1j * (
        np.tile(pm['K_ps'], (1, n_ifg)) * bperp_mat +
        np.tile(pm['C_ps'], (1, n_ifg))
    )))

    ifg_std = np.sqrt(np.sum(ph_diff**2, axis=0) / n_ps) * 180 / np.pi

    for i in range(n_ifg):
        date_str = datetime.datetime.fromtimestamp(int(ps['day'][i] * 86400)).strftime("%d-%b-%Y")
        print(f"{i+1:3d} {date_str} {ifg_std[i]:.2f}")
    
    save_h5(workdir,ifgstdname,**{'ifg_std':ifg_std})

def step_5_ps_merge(workdir:str,parms:dict):
    """
    Extended Python translation of the MATLAB step_5_ps_merge_patches(psver).
    This code includes the logic for:
      - intersection (C, IA, IB) if grid_size == 0
      - weighted merges if grid_size != 0
      - overlap-difference merges (ph_uw_diff, etc.) for ph_uw, scla, scn
      - final sorting, duplicate removal, and variable saving
    Portions assume domain-specific functions: llh2local(), getparm(), etc.
    """
    print("Running Step-5 ...")
    print('Merging patches...')

    psver = 2
    psname = f"ps{psver}.h5"
    phname = f"ph{psver}.h5"
    rcname = f"rc{psver}.h5"
    pmname = f"pm{psver}.h5"
    bpname = f"bp{psver}.h5"
    laname = f"la{psver}.h5"
    hgtname = f"hgt{psver}.h5"

    patchfile = os.path.join(workdir, "patch.list")
    dirname = []
    if os.path.isfile(patchfile):
        with open(patchfile, "r") as f:
            for line in f:
                patchname = line.strip()
                if patchname:
                    dirname.append(patchname)

    remove_ix = np.array([], dtype=bool)
    ij = np.zeros((0, 2), dtype=int)
    lonlat = np.zeros((0, 2), dtype=float)
    ph = np.zeros((0, 0), dtype=float)
    ph_rc = np.zeros((0, 0), dtype=float)
    ph_reref = np.zeros((0, 0), dtype=float)
    ph_patch = np.zeros((0, 0), dtype=float)
    ph_res = np.zeros((0, 0), dtype=float)
    K_ps = np.zeros((0, 0), dtype=float)
    C_ps = np.zeros((0, 0), dtype=float)
    coh_ps = np.zeros((0, 0), dtype=float)
    bperp_mat = np.zeros((0, 0), dtype=np.float32)
    la = np.zeros((0, 0), dtype=float)
    hgt = np.zeros((0, 0), dtype=float)

    # Loop through patch directories
    for patch in dirname:
        print(f"   Merging patch {patch}")

        correct_phase(os.path.join(workdir,patch),psver)

        ps = read_h5(os.path.join(workdir,patch, psname))

        # Load patch information and identify points in the patch
        patch_ij = np.loadtxt(os.path.join(workdir,patch, 'patch_noover.in'))
        ix = (
            (ps['ij'][:, 1] >= patch_ij[2] - 1) & 
            (ps['ij'][:, 1] <= patch_ij[3] - 1) & 
            (ps['ij'][:, 2] >= patch_ij[0] - 1) & 
            (ps['ij'][:, 2] <= patch_ij[1] - 1))

        IA, IB = intersect_rows(ps['ij'][ix, 1:3], ij)
        remove_ix = np.concatenate([remove_ix, IB])
        IA, IB = intersect_rows(ps['ij'][:,1:3], ij)
        ix_ex = np.ones(ps['n_ps'], dtype=bool)
        ix_ex[IA] = False
        ix[ix_ex] = 1

        ij = np.concatenate([ij, ps['ij'][ix, 1:3]])
        lonlat = np.concatenate([lonlat, ps['lonlat'][ix, :]])

        # Load and append phase data
        if os.path.exists(os.path.join(workdir,patch, phname)):
            phin = read_h5(os.path.join(workdir,patch, phname))
            ph_w = phin['ph']
        elif 'ph' in ps:
            ph_w = ps['ph']

        if 'ph_w' in locals():
            ph = np.vstack(ph_w[ix, :]) if ph.size == 0 else np.vstack([ph, ph_w[ix, :]])

        rc = read_h5(os.path.join(workdir,patch, rcname))
        ph_rc = np.vstack(rc['ph_rc'][ix, :]) if ph_rc.size == 0 else np.vstack([ph_rc, rc['ph_rc'][ix, :]])
        ph_reref = np.vstack(rc['ph_reref'][ix, :]) if ph_reref.size == 0 else np.vstack([ph_reref, rc['ph_reref'][ix, :]])

        pm = read_h5(os.path.join(workdir,patch, pmname))
        ph_patch = np.vstack(pm['ph_patch'][ix, :]) if ph_patch.size == 0 else np.vstack([ph_patch, pm['ph_patch'][ix, :]])
        if 'ph_res' in pm:
            ph_res = np.vstack(pm['ph_res'][ix, :]) if ph_res.size == 0 else np.vstack([ph_res, pm['ph_res'][ix, :]])
        if 'K_ps' in pm:
            K_ps = np.vstack(pm['K_ps'][ix]) if K_ps.size == 0 else np.vstack([K_ps, pm['K_ps'][ix]])
        if 'C_ps' in pm:
            C_ps = np.vstack(pm['C_ps'][ix]) if C_ps.size == 0 else np.vstack([C_ps, pm['C_ps'][ix]])
        if 'coh_ps' in pm:
            coh_ps = np.vstack(pm['coh_ps'][ix]) if coh_ps.size == 0 else np.vstack([coh_ps, pm['coh_ps'][ix]])

        bp = read_h5(os.path.join(workdir,patch, bpname))
        bperp_mat = np.vstack(bp['bperp_mat'][ix, :]) if bperp_mat.size == 0 else np.vstack([bperp_mat, bp['bperp_mat'][ix, :]])
        
        if os.path.exists(os.path.join(workdir,patch, laname)):
            lain = read_h5(os.path.join(workdir,patch, laname))
            la = np.vstack(lain['la'][ix]) if la.size == 0 else np.vstack([la, lain['la'][ix]])

        if os.path.exists(os.path.join(workdir,patch, hgtname)):
            hgtin = read_h5(os.path.join(workdir,patch, hgtname))
            hgt = np.vstack(hgtin['hgt'][ix]) if hgt.size == 0 else np.vstack([hgt, hgtin['hgt'][ix]])

    # Loop Ended through patch directories
    
    n_ps_orig = ij.shape[0]
    keep_ix = np.ones(n_ps_orig, dtype=bool)
    keep_ix[remove_ix] = False
    lonlat_save = lonlat.copy()
    coh_ps_weed = coh_ps[keep_ix]
    lonlat = lonlat[keep_ix, :]
    
    # Find unique rows and duplicate indices
    _, unique_indices = np.unique(lonlat, axis=0, return_index=True)
    dups_indices = np.setdiff1d(np.arange(lonlat.shape[0]), unique_indices)
    keep_ix_num = np.where(keep_ix)[0]

    # Process duplicates
    for dup_index in dups_indices:
        dups_ix_weed = np.where(
            (lonlat[:, 0] == lonlat[dup_index, 0]) & 
            (lonlat[:, 1] == lonlat[dup_index, 1])
        )[0]
        max_coh_index = np.argmax(coh_ps_weed[dups_ix_weed])
        remove_indices = np.delete(dups_ix_weed, max_coh_index)
        keep_ix[keep_ix_num[remove_indices]] = False

    if len(dups_indices) > 0:
        lonlat = lonlat_save[keep_ix, :]
        print(f' {len(dups_indices)} pixels with duplicate lon/lat dropped\n')
    
    ############################################################
    
    ll0 = (np.max(lonlat, axis=0) + np.min(lonlat, axis=0)) / 2
    xy = llh2local(lonlat.T, ll0) * 1000

    heading = parms['heading']
    if heading is None or heading == '':
        heading = 0
    theta = (180 - heading) * np.pi / 180
    if theta > np.pi:
        theta -= 2 * np.pi

    rotm = np.array([[np.cos(theta), np.sin(theta)], 
                    [-np.sin(theta), np.cos(theta)]])
    xynew = rotm @ xy
    if (max(xynew[0,:]) - min(xynew[0,:])) < (max(xy[0,:]) - min(xy[0,:])) and \
    (max(xynew[1,:]) - min(xynew[1,:])) < (max(xy[1,:]) - min(xy[1,:])):
        xy = xynew
        print(f' Rotating xy by {theta*180/np.pi} degrees')
    
    xy = xy.T.astype(np.float32)
    sort_ix = np.lexsort((xy[:, 0], xy[:, 1]))
    xy_sort = xy[sort_ix]
    xy = np.column_stack((np.arange(1, len(xy_sort) + 1), xy_sort))
    xy[:, 1:3] = np.round(xy[:, 1:3] * 1000) / 1000
    lonlat = lonlat[sort_ix]
    
    all_ix = np.arange(ij.shape[0])
    keep_ix = all_ix[keep_ix]
    sort_ix = keep_ix[sort_ix]
    n_ps = len(sort_ix)
    print(f' Writing merged dataset (contains {n_ps} pixels)')

    ph_rc = ph_rc[sort_ix, :]
    non_zero_mask = ph_rc != 0
    ph_rc[non_zero_mask] /= np.abs(ph_rc[non_zero_mask])
    ph_reref = ph_reref[sort_ix, :]
    save_h5(workdir, rcname , **{'ph_rc': ph_rc, 'ph_reref': ph_reref})

    ph_patch = ph_patch[sort_ix, :]
    ph_res = ph_res[sort_ix, :] if ph_res.shape[0] == n_ps_orig else np.array([])
    K_ps = K_ps[sort_ix, :] if K_ps.shape[0] == n_ps_orig else np.array([])
    C_ps = C_ps[sort_ix, :] if C_ps.shape[0] == n_ps_orig else np.array([])
    coh_ps = coh_ps[sort_ix, :] if coh_ps.shape[0] == n_ps_orig else np.array([])
    save_h5(workdir, pmname , **{'ph_patch': ph_patch, 'ph_res': ph_res,
                                'K_ps': K_ps, 'C_ps': C_ps,"coh_ps":coh_ps})
        
    ph = ph[sort_ix, :] if ph.shape[0] == n_ps_orig else np.array([])
    save_h5(workdir, phname , **{'ph': ph})
    
    la = la[sort_ix, :] if la.shape[0] == n_ps_orig else np.array([])
    save_h5(workdir, laname , **{'la': la})
    
    hgt = hgt[sort_ix, :] if hgt.shape[0] == n_ps_orig else np.array([])
    save_h5(workdir, hgtname , **{'hgt': hgt})

    bperp_mat = bperp_mat[sort_ix, :]
    save_h5(workdir, bpname , **{'bperp_mat': bperp_mat})

    ps_new = ps.copy()
    ps_new['n_ps'] = n_ps
    ps_new['ij'] = np.column_stack((np.arange(1, n_ps + 1), ij[sort_ix, :]))
    ps_new['xy'] = xy
    ps_new['lonlat'] = lonlat
    save_h5(workdir, psname , **ps_new)

    save_h5(workdir, 'psver.h5' , **{'psver': psver})

    print("Estimating noise standard deviation (degrees)...")
    estimate_noise_stddev(workdir,psver)
