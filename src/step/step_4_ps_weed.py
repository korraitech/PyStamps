from typing import Optional
import numpy as np
import os
from typing import Optional
from logit import logit
from getparm import getparm
from stamps_save import stamps_save
from ps_est_gamma_quick import load_mat


def step_4_ps_weed(all_da_flag: int = 0,
            no_weed_adjacent: Optional[int] = None,
            no_weed_noisy: Optional[int] = None) -> None:
    """
    Python translation of the MATLAB ps_weed.m function.

    PS_WEED weeds out neighboring PS and saves those kept to a new version.
    
    References similarly used code/logic from ps_est_gamma_quick.py
    for missing dependencies like getparm, logit, stamps_save, etc.

    Args:
        all_da_flag (int): 0 or 1, indicating whether to also process "other" data.
        no_weed_adjacent (Optional[int]): If 0, weed adjacent pixels; if 1, skip. 
                                          If None, check weed_neighbours param.
        no_weed_noisy (Optional[int]): If 0, weed noisy pixels; if 1, skip.
                                       If None, uses standard_dev & max_noise thresholds.
    """

    logit("Weeding selected pixels...")

    # Retrieve necessary parameters from config
    weed_neighbours = getparm("weed_neighbours", "y")
    weed_standard_dev = getparm("weed_standard_dev", 1)
    weed_max_noise = getparm("weed_max_noise", 1)
    weed_zero_elevation = getparm("weed_zero_elevation", "n")
    drop_ifg_index = getparm("drop_ifg_index", [])
    small_baseline_flag = getparm("small_baseline_flag", "n")

    drop_ifg_index = np.array(drop_ifg_index[0]).flatten()

    # Decide adjacency-weeding if not explicitly set
    if no_weed_adjacent is None:
        no_weed_adjacent = 0 if weed_neighbours[0][0].lower() == 'y' else 1

    # Decide noise-weeding if not explicitly set
    if no_weed_noisy is None:
        # If thresholds are >= pi, skip noise weeding
        if weed_standard_dev[0][0][0] >= np.pi and weed_max_noise[0][0][0] >= np.pi:
            no_weed_noisy = 1
        else:
            no_weed_noisy = 0

    # Load psver
    try:
        psver_data = load_mat("psver.mat")
        psver = int(psver_data["psver"][0][0])
    except FileNotFoundError:
        psver = 1

    # Filenames based on psver
    psname = f"ps{psver}"
    pmname = f"pm{psver}"
    phname = f"ph{psver}"
    selectname = f"select{psver}"
    hgtname = f"hgt{psver}.mat"
    laname = f"la{psver}.mat"
    incname = f"inc{psver}.mat"
    bpname = f"bp{psver}.mat"

    # "other" file placeholders
    psothername = "ps_other"
    pmothername = "pm_other"
    selectothername = "select_other"
    hgtothername = "hgt_other"
    laothername = "la_other"
    incothername = "inc_other"
    bpothername = "bp_other"

    # Load ps, select (sl), and phase
    ps = load_mat(psname + ".mat")
    sl = load_mat(selectname + ".mat")

    try:
        phin = load_mat(phname + ".mat")
        ph = phin["ph"]
    except FileNotFoundError:
        ph = ps["ph"]

    # Build ifg_index
    all_ifgs = np.arange(1, ps["n_ifg"] + 1, dtype=int)
    ifg_index = np.setdiff1d(all_ifgs, drop_ifg_index)

    day = ps["day"]
    bperp = ps["bperp"]
    # master_day = ps["master_day"]  # might not be needed

    # sl can have keep_ix or just ix2
    if "keep_ix" in sl:
        keep_ix = sl["keep_ix"].astype(int) - 1  # converting from 1-based
        ix2 = sl["ix"][keep_ix]
        K_ps2 = sl["K_ps2"][keep_ix]
        C_ps2 = sl["C_ps2"][keep_ix]
        coh_ps2 = sl["coh_ps2"][keep_ix]
    else:
        ix2 = sl["ix2"].astype(int) - 1
        K_ps2 = sl["K_ps2"]
        C_ps2 = sl["C_ps2"]
        coh_ps2 = sl["coh_ps2"]

    ij2 = ps["ij"][ix2, :]
    xy2 = ps["xy"][ix2, :]
    ph2 = ph[ix2, :]
    lonlat2 = ps["lonlat"][ix2, :]

    # Load pm (which has ph_patch, possibly other fields)
    pm = load_mat(pmname + ".mat")
    ph_patch2 = pm["ph_patch"][ix2, :]

    # If sl has ph_res2, subset similarly
    if "ph_res2" in sl:
        ph_res_all = sl["ph_res2"]
        # If keep_ix was used
        if "keep_ix" in sl:
            ph_res2 = ph_res_all[keep_ix, :]
        else:
            ph_res2 = ph_res_all
    else:
        ph_res2 = np.array([])

    # Remove large arrays from ps to free memory
    if "ph" in ps:
        del ps["ph"]
    for key_rm in ["xy", "ij", "lonlat", "sort_ix"]:
        if key_rm in ps:
            del ps[key_rm]

    # Optional step for "other" data, if all_da_flag != 0
    n_ps_other = 0
    if all_da_flag != 0:
        pso = load_mat(psothername + ".mat")
        slo = load_mat(selectothername + ".mat")

        ix_other = slo["ix_other"].astype(bool)
        n_ps_other = np.sum(ix_other)

        K_ps_other2 = pso["K_ps_other"][ix_other]
        C_ps_other2 = pso["C_ps_other"][ix_other]
        coh_ps_other2 = pso["coh_ps_other"][ix_other]
        ph_res_other2 = pso["ph_res_other"][ix_other, :]

        ij2 = np.vstack([ij2, pso["ij_other"][ix_other, :]])
        xy2 = np.vstack([xy2, pso["xy_other"][ix_other, :]])
        ph2 = np.vstack([ph2, pso["ph_other"][ix_other, :]])
        lonlat2 = np.vstack([lonlat2, pso["lonlat_other"][ix_other, :]])

        pmo = load_mat(pmothername + ".mat")
        ph_patch_other2 = pmo["ph_patch_other"][ix_other, :]

        K_ps2 = np.concatenate([K_ps2, K_ps_other2])
        C_ps2 = np.concatenate([C_ps2, C_ps_other2])
        coh_ps2 = np.concatenate([coh_ps2, coh_ps_other2])
        ph_patch2 = np.vstack([ph_patch2, ph_patch_other2])
        if ph_res2.size == 0:
            ph_res2 = ph_res_other2
        else:
            ph_res2 = np.vstack([ph_res2, ph_res_other2])

    # Load hgt if it exists
    try:
        hgt_dict = load_mat(hgtname)
        hgt = hgt_dict["hgt"][ix2]
        if all_da_flag != 0:
            hto = load_mat(hgtothername)
            hgt = np.concatenate([hgt, hto["hgt_other"][ix_other]])
    except FileNotFoundError:
        hgt = None

    n_ps_low_D_A = len(ix2)
    n_ps = n_ps_low_D_A + n_ps_other
    ix_weed = np.ones(n_ps, dtype=bool)

    msg_ps = f"{n_ps_low_D_A} low D_A PS, {n_ps_other} high D_A PS"
    logit(msg_ps)

    # -----------------------------------------------------
    # 1) Weeding adjacent pixels if no_weed_adjacent == 0
    # -----------------------------------------------------
    if no_weed_adjacent == 0:
        logit("Weeding adjacent pixels (equivalent to 'drop neighbors') ...")
        # The original MATLAB code creates a 3×3 region around each pixel and merges them,
        # then keeps the pixel with the highest coherence.
        # Here is a placeholder approach. You might replicate the full logic or adapt:
        ij_shift = ij2[:, 1:3] if ij2.shape[1] > 2 else ij2[:, 0:2]
        # (The MATLAB code has e.g. ij2(:,2:3) => depends on shape. Adjust as needed.)
        
        # Shift the indices so everything is >=0
        mn = ij_shift.min(axis=0)
        ij_shift = ij_shift - mn + 2  # offset by 2 so we can do a +/-1 window
        max_i = ij_shift[:, 0].max() + 2
        max_j = ij_shift[:, 1].max() + 2

        neigh_ix = np.zeros((max_i, max_j), dtype=int)
        # Mark neighbors
        miss_middle_mask = np.ones((3, 3), dtype=bool)
        miss_middle_mask[1, 1] = False

        n_ps_total = n_ps
        for i_ps in range(n_ps_total):
            px = ij_shift[i_ps, 0]
            py = ij_shift[i_ps, 1]
            xlow, xhigh = px - 1, px + 2
            ylow, yhigh = py - 1, py + 2
            subarray = neigh_ix[xlow:xhigh, ylow:yhigh]
            # We only fill zeros with current i_ps+1 if miss_middle_mask is True
            fill_mask = (subarray == 0) & miss_middle_mask
            subarray[fill_mask] = i_ps + 1  # store 1-based
            neigh_ix[xlow:xhigh, ylow:yhigh] = subarray

            if (i_ps + 1) % 100000 == 0:
                logit(f"{i_ps+1} PS processed for adjacency mapping", 2)

        # Next, build adjacency list
        neigh_ps = [[] for _ in range(n_ps_total)]
        for i_ps in range(n_ps_total):
            px = ij_shift[i_ps, 0]
            py = ij_shift[i_ps, 1]
            myid = neigh_ix[px, py]
            if myid != 0:
                # means there's a 'master' ID at (px, py), it might have references
                # We'll store i_ps in the adjacency list of 'myid-1' (0-based)
                neigh_ps[myid - 1].append(i_ps)

        # Now, merge sets that share adjacency
        for i_ps in range(n_ps_total):
            if neigh_ps[i_ps]:
                same_ps_stack = [i_ps]
                idx = 0
                while idx < len(same_ps_stack):
                    ps_i = same_ps_stack[idx]
                    same_ps_stack += neigh_ps[ps_i]
                    neigh_ps[ps_i] = []
                    idx += 1
                same_ps_stack = np.unique(same_ps_stack)
                # pick the pixel among same_ps_stack with highest coherence
                sub_cohs = coh_ps2[same_ps_stack]
                best_ix = np.argmax(sub_cohs)
                # keep that one => drop the rest in ix_weed
                for i_sp, sp_ind in enumerate(same_ps_stack):
                    if i_sp != best_ix:
                        ix_weed[sp_ind] = False

        logit(f"{np.sum(ix_weed)} PS kept after dropping adjacent pixels")

    # -----------------------------------------------------
    # 2) Weeding zero-elevation or near-zero if required
    # -----------------------------------------------------
    if weed_zero_elevation.lower() == 'y':
        # The MATLAB code does: if hgt=0 => drop
        # Make sure we have hgt loaded, else skip
        if hgt is not None:
            hgt_weed = hgt[ix_weed]
            zero_mask = np.isclose(hgt_weed, 0)
            # drop them
            ill_idx = np.where(ix_weed)[0][zero_mask]
            ix_weed[ill_idx] = False
            logit(f"{np.sum(zero_mask)} pixels with zero elevation dropped.")
        else:
            logit("Warning: weed_zero_elevation='y', but no hgt available. Skipping zero-elev weed.")

    # current count
    n_ps_kept = np.sum(ix_weed)
    logit(f"{n_ps_kept} total PS so far (after adjacency & zero-elev weeding).")

    # -----------------------------------------------------
    # 3) If no_weed_noisy == 0 => weed out noisy pixels
    # -----------------------------------------------------
    if n_ps_kept > 0 and no_weed_noisy == 0:
        logit("Dropping noisy pixels...")

        # Subset to current set
        ix_weed_indices = np.where(ix_weed)[0]
        ph_weed = ph2[ix_weed, :].astype(np.complex128)

        # If small-baseline => shape might be (n_ps, n_ifg). Possibly subtract range error, etc.
        # The MATLAB code does: ph_weed = ph_weed * exp(-j*(K_ps2(ix_weed)*bperp'))
        # We'll do a minimal approach if bperp is 1D or 2D.
        # This can get complicated if bperp is shaped [n_ps, n_ifg].
        # For demonstration, assume bperp is 1D:
        if bperp.ndim == 1 and bperp.size == ph_weed.shape[1]:
            ph_weed *= np.exp(-1j * np.outer(K_ps2[ix_weed], bperp))
        ph_weed /= np.maximum(np.abs(ph_weed), 1e-12)

        if small_baseline_flag.lower() != 'y':
            # Add master noise => ph_weed[:, master_ix] = exp(j*C_ps2(ix_weed))
            master_ix = int(ps["master_ix"]) - 1 if "master_ix" in ps else 0
            if 0 <= master_ix < ph_weed.shape[1]:
                # shape mismatch possible if we lost an ifg in ps_est_gamma_quick
                ph_weed[:, master_ix] = np.exp(1j * C_ps2[ix_weed])

        # Delaunay approach for adjacency => get edges
        xy_sub = xy2[ix_weed, :]
        if xy_sub.shape[1] > 2:
            xy_sub = xy_sub[:, 1:3]  # approximate
        from scipy.spatial import Delaunay
        tri = Delaunay(xy_sub)
        edges = set()
        for simplex in tri.simplices:
            s0, s1, s2 = simplex
            edges.add(tuple(sorted((s0, s1))))
            edges.add(tuple(sorted((s1, s2))))
            edges.add(tuple(sorted((s2, s0))))
        edges = np.array(list(edges), dtype=int)

        # For each edge, compute phase difference across ifg_index
        if len(edges) == 0:
            logit("No edges found in triangulation, skipping noisy weeding.")
        else:
            dph_space = ph_weed[edges[:, 1], :] * np.conjugate(ph_weed[edges[:, 0], :])
            # Subset columns by ifg_index (which is 1-based from MATLAB)
            dph_space = dph_space[:, ifg_index - 1] if ifg_index.size > 0 else dph_space

            # We'll do a simpler approach for standard dev & max phase difference
            dph_angles = np.angle(dph_space)
            edge_std = np.std(dph_angles, axis=1)
            edge_max = np.max(np.abs(dph_angles), axis=1)

            # For each edge => update ps_std, ps_max
            n_ps_sub = ph_weed.shape[0]
            ps_std = np.full(n_ps_sub, np.inf, dtype=np.float32)
            ps_max = np.full(n_ps_sub, np.inf, dtype=np.float32)
            for i_edg, (p1, p2) in enumerate(edges):
                es = edge_std[i_edg]
                em = edge_max[i_edg]
                if ps_std[p1] > es:
                    ps_std[p1] = es
                if ps_std[p2] > es:
                    ps_std[p2] = es
                if ps_max[p1] > em:
                    ps_max[p1] = em
                if ps_max[p2] > em:
                    ps_max[p2] = em

            # Compare with thresholds
            keep_mask = (ps_std < weed_standard_dev) & (ps_max < weed_max_noise)
            good_ix = ix_weed_indices[~keep_mask == False]  # convert local subset to global
            # Mark those not in keep_mask as weeded out
            to_drop = ix_weed_indices[~keep_mask]
            ix_weed[to_drop] = False

        n_ps_kept_final = np.sum(ix_weed)
        logit(f"{n_ps_kept_final} PS kept after dropping noisy pixels")

    # ---------------------------------------
    # 4) Save the no_ps_info => track if no PS
    # ---------------------------------------
    no_ps_info = {
        "stamps_step_no_ps": np.zeros((5, 1), dtype=np.int32)
    }
    n_left = np.sum(ix_weed)
    if n_left == 0:
        logit("***No PS points left after weeding.***")
        no_ps_info["stamps_step_no_ps"][3, 0] = 1  # matches MATLAB index=4 step

    save_mat("no_ps_info.mat", no_ps_info)

    # Build weed file (weedX) => store indices
    weedname = f"weed{psver}"
    stamps_save(weedname, ix_weed)

    # 5) Now reorganize arrays & save new pm, ph, ps, etc.
    new_coh_ps = coh_ps2[ix_weed]
    new_K_ps = K_ps2[ix_weed]
    new_C_ps = C_ps2[ix_weed]
    new_ph_patch = ph_patch2[ix_weed, :]
    if ph_res2.size > 0:
        new_ph_res = ph_res2[ix_weed, :]
    else:
        new_ph_res = ph_res2

    pmname_new = f"pm{psver + 1}"
    stamps_save(pmname_new, new_ph_patch, new_ph_res, new_coh_ps, new_K_ps, new_C_ps)

    # Next, ph
    ph2_subset = ph2[ix_weed, :]
    phname_new = f"ph{psver + 1}"
    stamps_save(phname_new, ph2_subset)

    # Rebuild ps struct
    xy2_subset = xy2[ix_weed, :]
    ij2_subset = ij2[ix_weed, :]
    lonlat2_subset = lonlat2[ix_weed, :]

    ps["xy"] = xy2_subset
    ps["ij"] = ij2_subset
    ps["lonlat"] = lonlat2_subset
    ps["n_ps"] = int(ph2_subset.shape[0])

    psname_new = f"ps{psver + 1}"
    save_mat(psname_new + ".mat", ps)

    # If hgt is loaded
    if hgt is not None:
        hgt_subset = hgt[ix_weed]
        hgtname_new = f"hgt{psver + 1}.mat"
        stamps_save(hgtname_new, hgt_subset)

    # If la
    try:
        la_data = load_mat(laname)
        la_arr = la_data["la"][ix2]
        if all_da_flag != 0:
            lao_data = load_mat(laothername)
            la_other = lao_data["la_other"]
            # apply ix_other => shape?
            # la_other = la_other[ix_other] if needed
            la_arr = np.concatenate([la_arr, la_other])
        la_arr = la_arr[ix_weed]
        laname_new = f"la{psver + 1}.mat"
        stamps_save(laname_new, la_arr)
    except FileNotFoundError:
        pass

    # If inc
    try:
        inc_data = load_mat(incname)
        inc_arr = inc_data["inc"][ix2]
        if all_da_flag != 0:
            inco_data = load_mat(incothername)
            inc_other = inco_data["inc_other"]
            # inc_other = inc_other[ix_other]
            inc_arr = np.concatenate([inc_arr, inc_other])
        inc_arr = inc_arr[ix_weed]
        incname_new = f"inc{psver + 1}.mat"
        stamps_save(incname_new, inc_arr)
    except FileNotFoundError:
        pass

    # If bp
    try:
        bp_data = load_mat(bpname)
        bperp_mat = bp_data["bperp_mat"][ix2, :]
        if all_da_flag != 0:
            bpo_data = load_mat(bpothername)
            bperp_mat_other = bpo_data["bperp_other"]
            # bperp_mat_other = bperp_mat_other[ix_other,:]
            bperp_mat = np.vstack([bperp_mat, bperp_mat_other])
        bperp_mat = bperp_mat[ix_weed, :]
        bpname_new = f"bp{psver + 1}.mat"
        stamps_save(bpname_new, bperp_mat)
    except FileNotFoundError:
        pass

    # Remove scla_smooth, scla, etc. for psver+1
    for prefix in [
        "scla_smooth", "scla", "aps", "scn", "scla_smooth_sb", "scla_sb"
    ]:
        fname = f"{prefix}{psver+1}.mat"
        if os.path.exists(fname):
            os.remove(fname)

    setpsver(psver + 1)  # increment version
    logit("ps_weed completed successfully!", level=1)
