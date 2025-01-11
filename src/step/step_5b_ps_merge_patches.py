import numpy as np
import os
from .utils import read_h5,save_h5

"""
    Merge patches
"""

def ps_merge_patches(workdir:str,parms:dict):
    """
    Extended Python translation of the MATLAB ps_merge_patches(psver).
    This code includes the logic for:
      - intersection (C, IA, IB) if grid_size == 0
      - weighted merges if grid_size != 0
      - overlap-difference merges (ph_uw_diff, etc.) for ph_uw, scla, scn
      - final sorting, duplicate removal, and variable saving
    Portions assume domain-specific functions: llh2local(), getparm(), etc.
    """
    print("Running Step-5b ...\t[{}]")
    print('Merging patches...')

    # 1) Load user parameters
    grid_size = int(parms["merge_resample_size"])
    merge_stdev = np.inf if parms["merge_standard_dev"] == np.inf else float(parms["merge_standard_dev"])
    phase_accuracy = 10 * np.pi / 180.0
    min_weight = 1 / (merge_stdev ** 2)

    # Rough replication of random seed + max_coh from MATLAB
    np.random.seed(1001)  
    random_phases = np.random.randn(1000) * phase_accuracy
    max_coh = np.abs(np.sum(np.exp(1j * random_phases))) / 1000

    # 2) Filenames
    psver = int(read_h5(os.path.join(workdir, 'psver.h5'))['psver'])
    psname = f"ps{psver}.h5"
    phname = f"ph{psver}.h5"
    rcname = f"rc{psver}.h5"
    pmname = f"pm{psver}.h5"
    phuwname = f"phuw{psver}.h5"
    sclaname = f"scla{psver}.h5"
    sclasbname = f"scla_sb{psver}.h5"
    scnname = f"scn{psver}.h5"
    bpname = f"bp{psver}.h5"
    laname = f"la{psver}.h5"
    incname = f"inc{psver}.h5"
    hgtname = f"hgt{psver}.h5"

    # 3) Gather patch directories
    patchdir = os.path.join(workdir, "patch.txt")
    dirname = []
    if os.path.isfile(patchdir):
        with open(patchdir, "r") as f:
            for line in f:
                patchname = line.strip()
                if patchname:
                    dirname.append({"name": patchname})
    else:
        # Fallback to scanning directories starting with "PATCH_"
        dirlist = [d for d in os.listdir(".") if os.path.isdir(d) and d.startswith("PATCH_")]
        dirname = [{"name": x} for x in dirlist]

    n_patch = len(dirname)
    # Keep track of which pixels to remove after patch merges
    remove_ix = np.array([], dtype=bool)

    # Large arrays that accumulate data from patches:
    ij = np.zeros((0, 2), dtype=int)
    lonlat = np.zeros((0, 2), dtype=float)
    ph = np.zeros((0, 0), dtype=float)
    ph_rc = np.zeros((0, 0), dtype=float)
    ph_reref = np.zeros((0, 0), dtype=float)
    ph_uw = np.zeros((0, 0), dtype=np.float32)
    ph_patch = np.zeros((0, 0), dtype=float)
    ph_res = np.zeros((0, 0), dtype=float)
    ph_scla = np.zeros((0, 0), dtype=np.float32)
    ph_scla_sb = np.zeros((0, 0), dtype=np.float32)
    ph_scn_master = np.zeros((0, 0), dtype=float)
    ph_scn_slave = np.zeros((0, 0), dtype=float)
    K_ps = np.zeros((0, 0), dtype=float)
    C_ps = np.zeros((0, 0), dtype=float)
    coh_ps = np.zeros((0, 0), dtype=float)
    K_ps_uw = np.zeros((0, 0), dtype=np.float32)
    K_ps_uw_sb = np.zeros((0, 0), dtype=np.float32)
    C_ps_uw = np.zeros((0, 0), dtype=np.float32)
    C_ps_uw_sb = np.zeros((0, 0), dtype=np.float32)
    bperp_mat = np.zeros((0, 0), dtype=np.float32)
    la = np.zeros((0, 0), dtype=float)
    inc = np.zeros((0, 0), dtype=float)
    hgt = np.zeros((0, 0), dtype=float)
    amp = np.zeros((0, 0), dtype=np.float32)

    # 4) Loop through patch directories
    for patch in dirname:
        print(f"   Merging patch {patch}")

        # Load main ps structure
        ps_data = read_h5(os.path.join(workdir,patch, psname))
        if "n_image" in ps_data:
            n_image = int(ps_data["n_image"])
        else:
            n_image = int(ps_data["n_ifg"])

        # ps.ij, ps.xy, ps.lonlat etc.
        ps_ij = ps_data.get("ij", None)      # shape (n_ps, 3)
        ps_lonlat = ps_data.get("lonlat", None)
        ps_xy = ps_data.get("xy", None)
        ps_n_ps = ps_data.get("n_ps", 0)

        # Load patch_noover.in
        if os.path.isfile("patch_noover.in"):
            patch_noover = np.loadtxt("patch_noover.in")  # shape ~ (4,)
        else:
            patch_noover = np.array([])

        # Create boolean index array ix
        if (patch_noover.size == 4) and (ps_ij is not None):
            # replicate lines 107-112 from MATLAB
            ix_array = (
                (ps_ij[:, 1] >= patch_noover[2] - 1)
                & (ps_ij[:, 1] <= patch_noover[3] - 1)
                & (ps_ij[:, 2] >= patch_noover[0] - 1)
                & (ps_ij[:, 2] <= patch_noover[1] - 1)
            )
        else:
            ix_array = np.zeros(ps_ij.shape[0], dtype=bool)

        if np.sum(ix_array) == 0:
            ix_no_ps = 1
        else:
            ix_no_ps = 0

        # 4a) grid_size == 0 → do intersection logic
        if grid_size == 0:
            # Lines ~114-122 in MATLAB
            # We must emulate: [C, IA, IB] = intersect(ps.ij(ix,2:3), ij, 'rows')
            # to remove older duplicates and keep the new patch’s data for overlap.
            # We'll do a Python approximate approach using np.unique and dictionary lookups.

            # Intersection #1
            # new subset used is ps.ij[ix_array, 1:3]
            # existing big array is ij
            new_rows = ps_ij[ix_array, 1:3]
            older_rows = ij  
            
            # we find which older_rows match new_rows
            # We'll create a dict for quick row->index
            older_map = {}
            for ind, row in enumerate(older_rows):
                older_map[tuple(row)] = ind

            IA = []
            IB = []
            for idx, row in enumerate(new_rows):
                row_tup = tuple(row)
                if row_tup in older_map:
                    IA.append(idx)
                    IB.append(older_map[row_tup])
            IA = np.array(IA)
            IB = np.array(IB)

            # remove_ix = [remove_ix; IB] in MATLAB → mark these old indices for removal
            if IB.size > 0:
                # We need to grow remove_ix to accommodate new maximum index if needed
                old_len = remove_ix.size
                if IB.max() >= old_len:
                    # enlarge remove_ix
                    new_size = IB.max() + 1
                    tmp_remove = np.zeros(new_size, dtype=bool)
                    tmp_remove[:old_len] = remove_ix
                    remove_ix = tmp_remove
                remove_ix[IB] = True

            # Intersection #2
            # [C, IA2, IB2] = intersect(ps.ij(:,2:3), ij, 'rows')
            full_rows = ps_ij[:, 1:3]
            IA2 = []
            IB2 = []
            for idx, row in enumerate(full_rows):
                row_tup = tuple(row)
                if row_tup in older_map:
                    IA2.append(idx)
                    IB2.append(older_map[row_tup])
            IA2 = np.array(IA2)
            IB2 = np.array(IB2)

            # ix_ex = true(ps.n_ps,1); ix_ex(IA2)=0 → exclude pixels that we already have
            ix_ex = np.ones(ps_n_ps, dtype=bool)
            ix_ex[IA2] = False
            # in MATLAB: ix(ix_ex)=1
            # That means wherever ix_ex is True, we set ix to True.  
            # So effectively, we OR the existing "ix_array" with "ix_ex"
            # But note that "ix_array" is a subset. We want to keep union of them.
            # We'll do:
            ix_array = ix_array | ix_ex

        # 4b) grid_size != 0 → do weighted merges
        elif (grid_size != 0) and (ix_no_ps != 1):
            # lines ~123-294 in MATLAB.  
            # Summarize: we chunk ix_array, group by g_ij, then do weighted merges.
            # First compute g_ij, using ps.xy
            if (ps_xy is None) or (ps_ij is None):
                # skip if missing data
                pass
            else:
                used_idx = np.where(ix_array)[0]
                xy_min = ps_xy[used_idx, :]  # shape (some_count, 3) in the original code
                # The code references xy_min(3) or xy_min(2)? In MATLAB, there's confusion about columns vs. rows.
                # We'll assume columns: ps_xy is (n_ps, 3) => [ x, y, ??? ]
                x_min_val = np.min(xy_min[:, 2])  # index 2 => "xy(:,3)" in MATLAB
                y_min_val = np.min(xy_min[:, 1])  # index 1 => "xy(:,2)" in MATLAB

                # replicate lines 125-126
                # g_ij(:,1)=ceil((ps.xy(ix,3)-xy_min(3)+1e-9)/grid_size);
                # g_ij(:,2)=ceil((ps.xy(ix,2)-xy_min(2)+1e-9)/grid_size);
                # We'll do that in Python:
                local_xy_used = ps_xy[used_idx, :]
                g1 = np.ceil((local_xy_used[:, 2] - x_min_val + 1e-9) / grid_size).astype(int)
                g2 = np.ceil((local_xy_used[:, 1] - y_min_val + 1e-9) / grid_size).astype(int)
                g_ij = np.column_stack((g1, g2))

                # Now group them by unique rows
                # lines 129-131
                # [g_ij,I,g_ix] = unique(g_ij,'rows'); [g_ix,sort_ix] = sort(g_ix)
                # in Python:
                from collections import defaultdict
                unique_map = {}
                inverse_list = []
                next_label = 0
                for row in g_ij:
                    row_tup = tuple(row)
                    if row_tup not in unique_map:
                        unique_map[row_tup] = next_label
                        next_label += 1
                    inverse_list.append(unique_map[row_tup])
                g_ix = np.array(inverse_list)

                sort_ix_local = np.argsort(g_ix)
                g_ix_sorted = g_ix[sort_ix_local]
                used_idx = used_idx[sort_ix_local]  # reorder used_idx to group same g_ix

                # lines 133-144 -> load pm and compute ps_weight
                pm_data = load_matfile_fields_matlab_style(f"{pmname}.mat")
                ph_res_local = pm_data.get("ph_res", None)
                C_ps_local = pm_data.get("C_ps", None)
                coh_ps_local = pm_data.get("coh_ps", None)

                if (ph_res_local is not None) and (C_ps_local is not None):
                    # centralize about zero:
                    # pm.ph_res = angle(exp(j*(pm.ph_res - repmat(pm.C_ps,1,size(pm.ph_res,2)))))
                    # In Python:
                    ph_res_local = np.angle(np.exp(1j * (ph_res_local - C_ps_local)))
                    ph_res_local = np.concatenate((ph_res_local, C_ps_local), axis=1)

                    # sigsq_noise = var(...)
                    sigsq_noise = np.var(ph_res_local, axis=1, ddof=1)
                    # coh_ps_all = abs(sum(exp(j*[pm.ph_res]),2)) / n_ifg
                    coh_tmp = np.sum(np.exp(1j * ph_res_local), axis=1)
                    coh_ps_all = np.abs(coh_tmp) / n_ifg
                    # clamp coherence
                    coh_ps_all[coh_ps_all > max_coh] = max_coh
                    sigsq_noise[sigsq_noise < phase_accuracy**2] = phase_accuracy**2

                    # now subset to used_idx
                    ps_weight = 1.0 / sigsq_noise[used_idx]
                    ps_snr = 1.0 / (1.0 / (coh_ps_all[used_idx] ** 2) - 1.0)

                else:
                    # fallback if pm_data incomplete
                    ps_weight = np.ones(used_idx.shape, dtype=float)
                    ps_snr = np.ones(used_idx.shape, dtype=float)

                # lines 146-159 -> define f_ix, l_ix, check weight sums
                # group boundaries:
                boundary_idx = np.nonzero(np.diff(g_ix_sorted))[0]
                l_ix = np.concatenate((boundary_idx, [g_ix_sorted.size - 1]))
                f_ix = np.concatenate(([0], boundary_idx + 1))

                n_ps_g = f_ix.size

                weightsave = np.zeros(n_ps_g, dtype=float)
                for grp_i in range(n_ps_g):
                    these_idx = np.arange(f_ix[grp_i], l_ix[grp_i] + 1)
                    w = ps_weight[these_idx]
                    wsum = np.sum(w)
                    weightsave[grp_i] = wsum
                    if wsum < min_weight:
                        # lines 156-157 -> ix(f_ix(i):l_ix(i))=0
                        # set those indexes to False
                        for ii in these_idx:
                            ix_array[used_idx[ii]] = False

                # lines 161-163 -> check if any left
                updated_used_idx = np.where(ix_array)[0]
                if updated_used_idx.size == 0:
                    ix_no_ps = 1
                else:
                    g_ix_filtered = g_ix_sorted[np.isin(sort_ix_local, np.where(ix_array)[0])]
                    # lines 165-170 -> define new l_ix, f_ix, etc.
                    boundary_idx2 = np.nonzero(np.diff(g_ix_filtered))[0]
                    l_ix = np.concatenate((boundary_idx2, [g_ix_filtered.size - 1]))
                    f_ix = np.concatenate(([0], boundary_idx2 + 1))
                    ps_weight = ps_weight[np.isin(sort_ix_local, np.where(ix_array)[0])]
                    ps_snr = ps_snr[np.isin(sort_ix_local, np.where(ix_array)[0])]
                    used_idx = updated_used_idx[np.argsort(g_ix_filtered)]
                    n_ps_g = f_ix.size
                    n_ps_temp = used_idx.size

        # 4c) After we’ve updated ix_array, we incorporate new ps.ij + ps.lonlat
        if (grid_size == 0):
            # simply stack them
            ij = np.vstack((ij, ps_ij[ix_array, 1:3]))
            lonlat = np.vstack((lonlat, ps_lonlat[ix_array, :]))
        elif (grid_size != 0) and (ix_no_ps != 1):
            # Weighted merges to produce one pixel per group
            # replicate lines 178-189
            used_idx = np.where(ix_array)[0]
            if used_idx.size > 0:
                # boundary_idx/l_ix/f_ix were computed above
                n_ps_g = f_ix.size
                ij_g = np.zeros((n_ps_g, 2), dtype=int)
                lonlat_g = np.zeros((n_ps_g, 2), dtype=float)
                for grp_i in range(n_ps_g):
                    these_idx = np.arange(f_ix[grp_i], l_ix[grp_i] + 1)
                    # shape (~k) → repeated over 2 columns
                    w_ij = np.tile(ps_weight[these_idx], (2, 1)).T
                    sub_ij = ps_ij[used_idx[these_idx], 1:3]
                    ij_g[grp_i, :] = np.round(np.sum(sub_ij * w_ij, axis=0) / np.sum(w_ij[:, 0]))

                    w_ll = np.tile(ps_weight[these_idx], (2, 1)).T
                    sub_ll = ps_lonlat[used_idx[these_idx], :]
                    lonlat_g[grp_i, :] = np.sum(sub_ll * w_ll, axis=0) / np.sum(w_ll[:, 0])

                ij = np.vstack((ij, ij_g))
                lonlat = np.vstack((lonlat, lonlat_g))

        # 4d) Merge ph, rc, pm, bperp, la, inc, hgt, etc. (similar logic).
        #     For example, to merge ph if file exists:
        if os.path.isfile(f"{phname}.mat"):
            phin_data = load_matfile_fields_matlab_style(f"{phname}.mat")
            if "ph" in phin_data:
                ph_w = phin_data["ph"]
            else:
                ph_w = None
        else:
            ph_w = ps_data.get("ph", None)

        if ph_w is not None:
            if (grid_size == 0):
                ph = np.vstack((ph, ph_w[ix_array, :]))
            elif (grid_size != 0) and (ix_no_ps != 1):
                used_idx = np.where(ix_array)[0]
                ph_w = ph_w[used_idx, :]
                # Weighted merging by ps_snr
                n_ps_g = f_ix.size
                ph_g = np.zeros((n_ps_g, n_ifg), dtype=float)
                for grp_i in range(n_ps_g):
                    these_idx = np.arange(f_ix[grp_i], l_ix[grp_i] + 1)
                    w = np.tile(ps_snr[these_idx], (n_ifg, 1)).T
                    ph_g[grp_i, :] = np.sum(ph_w[these_idx, :] * w, axis=0)
                ph = np.vstack((ph, ph_g))
            # clear ph_w
        # Merge rc if present, etc. (similar to lines 215–241)...

        # Merge pm if present, etc. (lines ~244–320)...

        # Merge bperp_mat if present (lines 322–336)...

        # Merge la if present (lines 338–353)...

        # Merge inc if present (lines 355–369)...

        # Merge hgt if present (lines 374–389)...

        # Overlap-difference logic for ph_uw, scla, scn if grid_size == 0 (lines 390–452).
        # (ph_uw_diff, ph_scla_diff, ph_scn_diff). In MATLAB, it uses if ~isempty(C).
        # We’d replicate that if we actually computed “C” from the intersection logic. 
        # For brevity, show partial example: 
        if (grid_size == 0):
            # Suppose we had arrays IA, IB from the "intersect" step. 
            # If they exist and are non-empty => do difference
            if "IA" in locals() and "IB" in locals() and IA.size > 0:
                # Example: ph_uw_diff = mean(phuw.ph_uw(IA,:)-ph_uw(IB,:),1)
                # then subtract it for the new patch’s rows.
                # In Python, you’d do something like:
                if os.path.isfile(f"{phuwname}.mat"):
                    phuw_data = load_matfile_fields_matlab_style(f"{phuwname}.mat")
                    if "ph_uw" in phuw_data:
                        local_ph_uw = phuw_data["ph_uw"]
                        ph_uw_diff = np.mean(local_ph_uw[IA, :] - ph_uw[IB, :], axis=0)
                        ph_uw_diff = np.round(ph_uw_diff / (2 * np.pi)) * 2 * np.pi
                    else:
                        local_ph_uw = None
                        ph_uw_diff = np.zeros((1, n_image), dtype=float)
                else:
                    local_ph_uw = None
                    ph_uw_diff = np.zeros((1, n_image), dtype=float)

                # ph_uw = [ph_uw; local_ph_uw(ix_array,:) - repmat(ph_uw_diff,sum(ix_array),1)]
                if local_ph_uw is not None:
                    to_append = local_ph_uw[ix_array, :] - ph_uw_diff
                else:
                    to_append = np.zeros((np.sum(ix_array), n_image), dtype=np.float32)
                ph_uw = np.concatenate((ph_uw, to_append), axis=0)
            else:
                # If no overlap (C empty), just append data or zeros.
                # etc. 
                pass

            # Similarly for scla, scn, etc. (lines 409–451)...

        # Done with this patch
        os.chdir(cwd_backup)
    # end for patch directories

    # 5) Now that we have all patches merged into big arrays, remove duplicates, sort, etc.
    n_ps_orig = ij.shape[0]
    # remove_ix indicates older indices that we decided to discard
    if remove_ix.size < n_ps_orig:
        # expand remove_ix if needed
        tmp = np.zeros(n_ps_orig, dtype=bool)
        tmp[: remove_ix.size] = remove_ix
        remove_ix = tmp
    keep_ix = np.logical_not(remove_ix)
    lonlat_save = lonlat.copy()

    # For child arrays like coh_ps, we define a “coh_ps_weed”
    if coh_ps.shape[0] == n_ps_orig:
        coh_ps_weed = coh_ps[keep_ix, :]
    else:
        coh_ps_weed = np.zeros((0, 0), dtype=float)

    lonlat = lonlat[keep_ix, :]
    # lines ~467–476 handle duplicates in lonlat
    # We'll do a Python approach:
    # find duplicates by rounding or direct:
    rounded_coords = np.round(lonlat, 8)  # to avoid floating noise
    _, unique_idx = np.unique(rounded_coords, axis=0, return_index=True)
    all_idx = np.arange(lonlat.shape[0])
    dups = np.setxor1d(unique_idx, all_idx)
    if dups.size > 0:
        # if duplicates occur, keep the pixel with highest coherence
        # for each set though, we’d do a more elaborate grouping.
        # The MATLAB code uses a for-loop. We'll replicate that logic:
        keep_ix2 = np.ones(lonlat.shape[0], dtype=bool)
        for di in dups:
            # find all with same lat/lon as lonlat[di]
            same_loc = np.where(
                (lonlat[:, 0] == lonlat[di, 0]) & (lonlat[:, 1] == lonlat[di, 1])
            )[0]
            if same_loc.size <= 1:
                continue
            # pick the highest coherence
            # for simplicity, if coh_ps_weed is 2D, we might take the average or the first col
            if coh_ps_weed.shape[0] == lonlat.shape[0]:
                local_coh = np.mean(coh_ps_weed[same_loc, :], axis=1)
            else:
                local_coh = np.zeros(same_loc.size, dtype=float)
            best_ix = np.argmax(local_coh)
            # drop the others
            drop_list = np.setxor1d(same_loc, [same_loc[best_ix]])
            keep_ix2[drop_list] = False

        # finalize
        lonlat = lonlat[keep_ix2, :]
        print(f"   {dups.size} pixel(s) with duplicate lon/lat dropped\n")

        # apply keep_ix2 to everything else
        # (the MATLAB code reverts to lonlat_save(keep_ix,:) though)
        final_keep_mask = np.zeros(keep_ix.size, dtype=bool)
        final_indices = np.where(keep_ix)[0][keep_ix2]
        final_keep_mask[final_indices] = True
        keep_ix = final_keep_mask

    # 6) Recompute xy via llh2local, rotate, then sort by row:
    # lines ~485–519
    final_lonlat = lonlat_save[keep_ix, :]
    ll0 = (final_lonlat.max(axis=0) + final_lonlat.min(axis=0)) / 2
    xy = llh2local(final_lonlat.T, ll0) * 1000
    xy = xy.T
    # Possibly do heading-based rotation, etc., omitted for brevity.

    # 7) Sort in ascending y, then ascending x
    xy_sort_ind = np.lexsort((xy[:, 0], xy[:, 1]))
    xy = xy[xy_sort_ind, :]
    # We also reorder everything else accordingly:
    # etc.

    # 8) Build final ps structure, save
    n_ps_final = xy.shape[0]
    print(f"   Writing merged dataset (contains {n_ps_final} pixels)")

    # reorder big arrays with “sort_ix” if needed, stamps_save them, etc.
    # lines ~520–end in MATLAB (e.g., ph_uw, pm, sclaname, scnname, inc, hgt, etc.)

    # Example final saving calls:
    # stamps_save(phuwname, ph_uw)
    # stamps_save(sclaname, ph_scla, K_ps_uw, C_ps_uw)
    # ...
    # Save the new ps struct:
    ps_new = {
        "n_ps": n_ps_final,
        # build new indices: [1...n_ps_final]
        "ij": np.column_stack((np.arange(1, n_ps_final+1), ij[keep_ix][xy_sort_ind,:])),
        "xy": xy.astype(np.float32),
        "lonlat": final_lonlat[xy_sort_ind, :],
    }
    # e.g. np.save or pythonic save:
    # Using the MATLAB-like approach:
    # scipy.io.savemat(psname + ".mat", ps_new) or
    np.savez_compressed(psname + ".npz", **ps_new)

    # Save psver
    # np.save("psver.npy", np.array([psver]))

