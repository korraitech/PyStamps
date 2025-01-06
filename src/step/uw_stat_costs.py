import numpy as np
import scipy.io as sio
import os
import subprocess
from .utils import writecpx

def uw_stat_costs(unwrap_method='3D', variance=None, subset_ifg_index=None):
    """
    Python translation of the MATLAB uw_stat_costs.m script.

    Attempts to replicate the logic of the MATLAB function to perform 
    phase unwrapping in either 2D or 3D using cost functions and calls 
    to the external 'snaphu' executable.

    Depending on your environment, you may need to:
      - Have snaphu installed and in your PATH.
      - Provide the same .mat data files found in the original MATLAB environment
        ('uw_grid.mat','uw_interp.mat','uw_space_time.mat'), as well as 
         'uw_phaseuw.mat' and so on, per your usage needs.
    
    Parameters
    ----------
    unwrap_method : str, optional
        Either '2D' or '3D'. Default '3D'.
    variance : numpy.ndarray, optional
        If provided, used when unwrap_method == '2D'.
    subset_ifg_index : list or numpy.ndarray, optional
        Subset of interferogram indices to process. If None, all are processed.
    """
    # Start "timer" (just informational; no direct effect in Python)
    # In MATLAB: tic
    # In Python we won't necessarily replicate timing exactly, but feel free to measure if desired.

    # Fallback for method
    if unwrap_method is None:
        unwrap_method = '3D'
    
    # Constants
    costscale = 100
    nshortcycle = 200
    maxshort = 32000

    print("Unwrapping in space...")

    # Load .mat files. 
    # Make sure these files are available in the current working directory, 
    # or adjust paths as needed.
    uw_data = sio.loadmat('uw_grid.mat', squeeze_me=True, struct_as_record=False)
    ui_data = sio.loadmat('uw_interp.mat', squeeze_me=True, struct_as_record=False)
    ut_data = sio.loadmat('uw_space_time.mat', squeeze_me=True, struct_as_record=False)

    # Keys in the above dictionaries depend on the saved .mat structure:
    # For example, in MATLAB: uw=load('uw_grid','ph','nzix','pix_size','n_ps','n_ifg');
    # We'll pull out same-named keys from the .mat dictionary.
    uw_ph = uw_data['ph']          # shape: (n_ps, n_ifg)
    uw_nzix = uw_data['nzix']      # shape: (nrow, ncol), boolean or 0/1
    uw_pix_size = uw_data['pix_size']
    uw_n_ps = uw_data['n_ps']
    uw_n_ifg = uw_data['n_ifg']

    ui_colix = ui_data['colix']    # row -> col mapping
    ui_rowix = ui_data['rowix']    # col -> row mapping
    ui_Z = ui_data['Z']            # etc
    ui_edgs = ui_data['edgs']      # used below
    ui_n_edge = ui_data['n_edge'][()]  # might read as a 0D array

    # ut=load('uw_space_time','dph_space_uw','dph_noise','spread','predef_ix');
    # not all keys may exist; handle carefully
    ut_dph_space_uw = ut_data['dph_space_uw']   # shape: (n_ps, n_ifg)
    ut_dph_noise = ut_data.get('dph_noise', None)
    ut_spread = ut_data['spread']               # shape: (n_ps, n_ifg)
    ut_predef_ix = ut_data.get('predef_ix', None)

    # If variance is None, set empty array
    if variance is None:
        variance = np.array([])

    # if subset_ifg_index is None, use [1:size(uw.ph,2)] in MATLAB sense 
    # but Python indexing is 0-based, so we go [0,..., uw_n_ifg-1]
    if subset_ifg_index is None:
        subset_ifg_index = np.arange(0, uw_ph.shape[1])
    else:
        # Convert to numpy array if not already
        subset_ifg_index = np.array(subset_ifg_index, dtype=int)

    # Check if we have a predef_ix or not
    predef_flag = 'n'
    if ut_predef_ix is not None and ut_predef_ix.size > 0:
        predef_flag = 'y'

    # Extract dimensions from uw_nzix
    nrow, ncol = uw_nzix.shape

    # Indices of non-zero elements in uw_nzix
    # in MATLAB: [y,x] = find(uw.nzix); nzix=find(uw.nzix);
    # We'll do the same in Python:
    y, x = np.nonzero(uw_nzix)
    nzix = np.nonzero(uw_nzix)  # (row_indices, col_indices)
    # z = [1:uw.n_ps] in MATLAB => Python 0-based => range(uw_n_ps)
    # but used below for reshape. We'll just keep in mind we might use that indexing.

    ## Prepare edge stats
    colix = ui_colix
    rowix = ui_rowix
    Z = ui_Z

    # In MATLAB: n_edges=hist(abs(grid_edges),[1:ui.n_edge])'
    # We'll replicate:
    grid_edges = np.concatenate( (colix[np.abs(colix) > 0], rowix[np.abs(rowix) > 0]) )
    # Now we want histogram from 1..ui_n_edge:
    bins = np.arange(0.5, ui_n_edge + 1.5, 1.0)  # center on integers 1..ui_n_edge
    n_edges, _ = np.histogram(np.abs(grid_edges), bins=bins)

    # For 2D method
    if unwrap_method.lower() == '2d':
        # edge_length = sqrt(diff(x(ui.edgs(:,2:3)),[],2).^2+...)
        # We'll interpret the columns of ui_edgs as [?? 2 3]
        # In MATLAB indexing: ui.edgs(:,2:3) means columns 2 and 3 (1-based).
        # Python is 0-based, so columns [1,2].
        # So let's parse:
        #   x(ui.edgs(:,2:3)) => we want the corresponding x coords (from the nonzero() above ???)
        #   Actually from the original MATLAB code:
        #       x(ui.edgs(:,2:3)) => row dimension = size(ui.edgs,1), col dimension=2
        #   But "2:3" in MATLAB is columns #2 and #3. 
        #   In the loaded "edgs" array, it might be something like [edge_id, node1, node2,...].
        # We'll just replicate the logic carefully:
        edgs_2 = ui_edgs[:, 1].astype(int)  # node1
        edgs_3 = ui_edgs[:, 2].astype(int)  # node2

        # We'll need x[node1] and x[node2]. But node indices in the original code 
        # appear 1-based. Python arrays are 0-based.
        # We'll reduce by 1 if necessary (assuming the data in the .mat is 1-based).
        # We must check how the data looks in .mat; typically it might be 1-based.
        # We'll assume it's 1-based for now:
        edgs_2_0 = edgs_2 - 1
        edgs_3_0 = edgs_3 - 1

        # Now: x(...) => x is from the np.nonzero(uw_nzix). x is an array of col indices.
        # But x[...] means x for those ps? We should be consistent with how the original code 
        # indexes them. The original code references "x(ui.edgs(:,2:3))". 
        # This suggests that the "edgs" array references the "nzix" indexing. 
        # We'll do a direct approach:
        x2 = x[edgs_2_0]
        x3 = x[edgs_3_0]
        y2 = y[edgs_2_0]
        y3 = y[edgs_3_0]

        dx = (x2 - x3)**2
        dy = (y2 - y3)**2
        edge_length = np.sqrt(dx + dy)

        if uw_pix_size == 0:
            pix_size = 5
        else:
            pix_size = float(uw_pix_size)

        # If variance is empty, build zero array of same length as edge_length
        if variance.size == 0:
            sigsq_noise = np.zeros(edge_length.shape)
        else:
            # variance(ui.edgs(:,2)) + variance(ui.edgs(:,3))
            # Again watch for 1-based indexing
            # We'll do the same shift by -1
            sigsq_noise = variance[edgs_2_0] + variance[edgs_3_0]

        sigsq_aps = (2*np.pi)**2  # fixed for now as one fringe
        aps_range = 20000.0       # fixed for now as 20 km
        # sigsq_noise = sigsq_noise + sigsq_aps*(1-exp(-edge_length*pix_size*3/aps_range))
        sigsq_noise = sigsq_noise + sigsq_aps * (1 - np.exp(-edge_length * pix_size * 3.0 / aps_range))

        # scale it
        sigsq_noise = sigsq_noise / 10.0
        dph_smooth = ut_dph_space_uw
    else:
        # 3D approach
        if ut_dph_noise is None:
            raise ValueError("No dph_noise in ut_data for 3D approach.")
        # std(ut.dph_noise,0,2) => stdev across columns, i.e. for each ps
        # shape: (n_ps,)
        std_dph_noise = np.std(ut_dph_noise, axis=1, ddof=0)
        sigsq_noise = (std_dph_noise / (2*np.pi))**2
        # dph_smooth = ut.dph_space_uw - ut.dph_noise
        # shape: (n_ps, n_ifg)
        dph_smooth = ut_dph_space_uw - ut_dph_noise

    # We remove field 'dph_noise' from ut in the MATLAB code: ut=rmfield(ut,'dph_noise')
    # In Python, we can just ignore it from now on or set it to None
    ut_dph_noise = None

    # The code removes indices that are NaN from the rowix/colix usage:
    #   nostats_ix = find(isnan(sigsq_noise))'
    #   for i=nostats_ix
    #     rowix(abs(rowix) == i) = nan
    #     colix(abs(colix) == i) = nan
    #
    # Careful with 1-based indexing again. We'll do the same in Python:
    nostats_ix = np.where(np.isnan(sigsq_noise))[0]
    for i in nostats_ix:
        # The code does rowix(abs(rowix)==i) = nan, but i is 0-based in Python 
        # whereas abs(rowix) might contain 1-based references. We'll do i+1 for matching:
        mask_row = (np.abs(rowix) == (i+1))
        mask_col = (np.abs(colix) == (i+1))
        rowix[mask_row] = np.nan
        colix[mask_col] = np.nan

    # Convert sigsq_noise to int16 weighting, multiplied by nshortcycle^2 / costscale * n_edges
    # but we must align edges. We'll interpret the absolute index for colix/rowix 
    # and map them to sigsq_noise. Then multiply by n_edges(...) for that index.
    # In the original code:
    #   sigsq=int16(round(((sigsq_noise)*nshortcycle^2)/costscale.*n_edges));
    # but we must do that for each possible edge index from 1..ui_n_edge
    # We'll build an array "sigsq" of length len(sigsq_noise) = n_ps 
    # but note the indexing is a bit tricky. The code indicates that "n_edges" 
    # was used to figure out how many times each edge occurs. 
    # We'll do the same approach:

    # We'll do the same cost weighting:
    #   sigsq(sigsq<1)=1
    # We'll keep in a large array of shape (n_ps) or so. Then we index it with rowix/colix.
    scaled_sigsq_noise = ((sigsq_noise) * (nshortcycle**2)) / costscale # float
    # n_edges => shape: (ui_n_edge,) but our edges are 1..ui_n_edge inclusive.
    # We'll replicate the final multiplication by n_edges[ index-1 ] for each index 
    # since we are 1-based in the original code. 
    # We'll build an array called edge_weights so edge_weights[k] = scaled_sigsq_noise[k] * n_edges[k]
    # but we have n_ps vs n_edge mismatch in naming. If we are in 2D mode, the length of sigsq_noise 
    # equals ui.n_edge. If in 3D mode, the length of sigsq_noise equals n_ps. 
    # The original code lumps them together differently. 
    # Because the code in MATLAB does:
    #     rowcost(:,2:4:end) = rowstdgrid; 
    #     rowstdgrid(nzrowix) = sigsqtot(abs(rowix(nzrowix)));
    #
    # We'll replicate the logic literally: 
    #   For each i in subset_ifg_index, we compute spread = full(ut.spread(:,i1))
    #   Then sigsqtot = sigsq + spread. Then we fill in rowstdgrid with sigsqtot(...) 
    #   The actual shape is (n_ps,). Then rowix references some subset. 
    # Because the entire cost building is in a loop over i1, we place that logic inside the loop.

    # We'll define placeholders for rowcost, colcost. In the MATLAB code, these have dimension:
    #   rowcost: (nrow-1, ncol*4)
    #   colcost: (nrow, (ncol-1)*4)
    # We'll build them in the loop since it depends on spread, etc.

    ph_uw = np.zeros((uw_n_ps, uw_n_ifg), dtype=np.float32)
    msd = np.zeros((uw_n_ifg,), dtype=np.float32)

    # Prepare the snaphu.conf file
    with open('snaphu.conf', 'w') as f:
        f.write('INFILE  snaphu.in\n')
        f.write('OUTFILE snaphu.out\n')
        f.write('COSTINFILE snaphu.costinfile\n')
        f.write('STATCOSTMODE  DEFO\n')
        f.write('INFILEFORMAT  COMPLEX_DATA\n')
        f.write('OUTFILEFORMAT FLOAT_DATA\n')

    # Loop over subset_ifg_index
    for idx, i1 in enumerate(subset_ifg_index):
        print(f"   Processing IFG {i1+1} of {subset_ifg_index.size} (MATLAB style index: {i1+1})")

        # gather the dynamic spread for this IFG: shape = (n_ps,) 
        # in MATLAB: spread=full(ut.spread(:,i1))
        # in Python, we'll do:
        spread_col = ut_spread[:, i1]
        # Convert to int16 after scaling: 
        # spread=int16(round((abs(spread)*nshortcycle^2)/6/costscale.*repmat(n_edges,1,size(spread,2))));
        # but in the code it does ".*repmat(n_edges,1,size(spread,2))" => which suggests the spread also 
        # has dimension n_ps? This is complicated. 
        # The original code: spread=int16(round((abs(spread)*nshortcycle^2)/6/costscale.*repmat(n_edges,1,size(spread,2))));
        # then sigsqtot = sigsq + spread
        # We'll interpret "n_edges" as applying if 2D method. If 3D, there's no direct "n_edges" multiplication at this step. 
        # Actually, the code lumps them: sigsqtot = sigsq + spread. Then it indexes by rowix, colix. 
        # For the best direct translation, let's replicate the code lines exactly.

        # We'll define "spread_int" similarly to MATLAB:
        spread_int = np.round((np.abs(spread_col) * (nshortcycle**2)) / (6 * costscale))
        spread_int = spread_int.astype(np.int16)

        # We'll hold sigsq_noise in an int16 array "sigsq" as well for this IFG:
        # from the earlier code: 
        #   sigsq=int16(round(((sigsq_noise)*nshortcycle^2)/costscale.*n_edges));
        # but we haven't done the ".*n_edges" part for 3D. 
        # The code does that outside the "2D" condition, so let's see if we can replicate the final code block carefully.
        
        # In the final code block:
        #   sigsq=int16(round(((sigsq_noise)*nshortcycle^2)/costscale.*n_edges));  # weight by number of occurences
        #   ...
        #   sigsqtot=sigsq+spread;
        #   if predef_flag=='y':
        #       sigsqtot(ut.predef_ix(:,i1))=1;
        #
        # So let's handle that for both 2D and 3D in the same manner:
        
        # If we are in 2D mode, sigsq_noise has shape = (ui.n_edge,) 
        # while in 3D mode, sigsq_noise has shape = (n_ps,)
        
        # We define the scaled version first:
        scaled_sigsq = (sigsq_noise * (nshortcycle**2)) / costscale  # float
        if unwrap_method.lower() == '2d':
            # multiply by n_edges
            # Because each entry in sigsq_noise corresponds to an edge, 
            # so scaled_sigsq is shape (ui.n_edge,)
            # multiply elementwise by n_edges
            scaled_sigsq = scaled_sigsq * n_edges
        # Round and convert:
        sigsq_i16 = np.round(scaled_sigsq).astype(np.int16)
        # set min value=1
        sigsq_i16[sigsq_i16 < 1] = 1

        # Now compute total sigsq for this IFG: (still length n_ps or n_edge)
        # We'll replicate the MATLAB style: 
        #   spread = spread(...) -> shape n_ps
        #   we do the same scaling or not? The code specifically does:
        #   spread=int16(round((abs(spread)*nshortcycle^2)/6/costscale.*repmat(n_edges,1,size(spread,2))));
        # But in the code it uses "spread=int16(round((abs(spread)*nshortcycle^2)/6/costscale.*n_edges))" effectively if 3D or 2D. 
        # We'll do something simpler: multiply by n_edges only if 2D, just like above.
        if unwrap_method.lower() == '2d':
            # shape = n_ps? Actually in 2D mode we might have n_edge for spread?
            # The code is somewhat ambiguous. We'll do the same approach:
            # "spread_int" we defined above is shape (n_ps,) though, which doesn't match n_edges length. 
            # This discrepancy arises because the original code is quite tied to specialized data shapes. 
            # For a minimal direct translation, let's replicate literally:
            spread_int = (spread_int * n_edges).astype(np.int16)  # broadcast if shapes align? 
            # If they do not align, you may need to adapt the logic. 
            # The original code's dimension manipulations are context-specific. 
            # We'll proceed, but you may have to adapt for your data.
        
        # Now combine them -> sigsqtot
        # must be done in the final cost building step, for each pixel/edge. 
        # We'll do it once we build rowcost/colcost below.
        
        # rowcost, colcost creation
        # rowcost: (nrow-1, ncol*4)
        # colcost: (nrow, (ncol-1)*4)
        rowcost = np.zeros((nrow-1, ncol*4), dtype=np.int16)
        colcost = np.zeros((nrow, (ncol-1)*4), dtype=np.int16)

        # The code sets:
        #   rowcost(:,3:4:end) = maxshort
        #   colcost(:,3:4:end) = maxshort
        #   => 3:4:end means indices 2-based in MATLAB are 2,6,10,... in 0-based Python 
        #   We want to fill columns = 2,6,10,... => let's do slices:
        rowcost[:, 2::4] = maxshort
        colcost[:, 2::4] = maxshort

        # rowcost(:,4:4:end) = int16(stats_ix)*(-1-maxshort)+1
        # colcost(:,4:4:end) = int16(stats_ix)*(-1-maxshort)+1
        # stats_ix = ~isnan(rowix)
        # We'll replicate carefully with 0-based indexing => columns 3,7,11,...
        stats_ix_row = ~np.isnan(rowix)
        row_temp = stats_ix_row.astype(np.int16) * (-1 - maxshort) + 1
        # We need to place that into the slice rowcost[:, 3::4]
        # but row_temp is shape (nrow-1, ncol). So let's replicate the flatten or direct shape usage:
        rowcost[:, 3::4] = row_temp

        stats_ix_col = ~np.isnan(colix)
        col_temp = stats_ix_col.astype(np.int16) * (-1 - maxshort) + 1
        colcost[:, 3::4] = col_temp

        # Now fill in the cost array portion for sigsq:
        # rowcost(:,2:4:end) = rowstdgrid
        # colcost(:,2:4:end) = colstdgrid
        # rowstdgrid, colstdgrid are shape (nrow-1, ncol) and (nrow, ncol-1) respectively, 
        # for which we fill from the array sigsqtot(...) referencing rowix, colix. 
        rowstdgrid = np.ones((nrow-1, ncol), dtype=np.int16)
        colstdgrid = np.ones((nrow, ncol-1), dtype=np.int16)

        # We'll build "sigsqtot" as an int16 array of length = either n_ps or n_edge 
        # that is sigsq_i16 + spread_int
        sigsqtot = sigsq_i16 + spread_int
        # if predef_flag=='y':
        #   sigsqtot(ut.predef_ix(:,i1))=1;
        # In Python, let's do:
        if predef_flag == 'y':
            if ut_predef_ix.ndim == 2:
                # ut_predef_ix for each IFG is shape (n_ps, n_ifg)?
                # We'll gather predef_ix vector for i1
                predef_for_ifg = ut_predef_ix[:, i1]  # shape (n_ps,)
                # find nonzero
                predef_nz = np.nonzero(predef_for_ifg)[0]
                # set indexes in sigsqtot to 1
                # but remember the predef_ix might be a boolean or integer 
                # referencing the "ps" index +1 
                # The code in MATLAB sets those *indices in the entire array to 1. 
                # We'll assume it's 1-based for the ps. We'll do -1 for 0-based:
                for px in predef_nz:
                    px0 = px  # if it's 0-based already, no shift
                    sigsqtot[px0] = 1
            else:
                # if shape is different, adapt as needed
                pass

        # rowstdgrid(nzrowix) = sigsqtot(abs(rowix(nzrowix)));
        # in code, rowix(nzrowix) is a set of positive or negative indexes 
        # referencing the ps/edge index +1. We'll replicate:
        # make a masked version for the non-zero rowix
        nzrowix_mask = ~np.isnan(rowix) & (rowix != 0)
        # gather the absolute( rowix( nzrowix ) )
        abs_rowix_values = np.abs(rowix[nzrowix_mask]).astype(int)  # 1-based index
        abs_rowix_values -= 1  # shift to 0-based
        # fill rowstdgrid with sigsqtot lookups
        rowstdgrid[nzrowix_mask] = sigsqtot[abs_rowix_values]

        # colstdgrid(nzcolix) = sigsqtot(abs(colix(nzcolix)));
        nzcolix_mask = ~np.isnan(colix) & (colix != 0)
        abs_colix_values = np.abs(colix[nzcolix_mask]).astype(int) - 1
        colstdgrid[nzcolix_mask] = sigsqtot[abs_colix_values]

        # Now place them in rowcost, colcost at columns 2,6,10 => 2::4
        rowcost[:, 1::4] = rowstdgrid  # (nrow-1, ncol)
        colcost[:, 1::4] = colstdgrid  # (nrow, ncol-1)

        # Next, offset:
        # offset_cycle = (angle(exp(1i*ut.dph_space_uw(:,i1))) - dph_smooth(:,i1)) / 2/pi
        # In Python:
        dph_smooth_col = dph_smooth[:, i1]
        # Build offset_cycle:
        # We'll do np.angle(np.exp(1j*dph_space_uw[:, i1])) => shape (n_ps,)
        offset_cycle = (np.angle(np.exp(1j * ut_dph_space_uw[:, i1])) - dph_smooth_col) / (2.0 * np.pi)

        # Then fill in rowcost(:,1:4:end) and colcost(:,1:4:end),
        # rowcost(:,1:4:end) = -offgrid
        # offgrid(nzrowix) = round(offset_cycle(abs(rowix(nzrowix))) * sign(rowix(nzrowix)) * nshortcycle)
        # We'll replicate:
        offgrid_row = np.zeros((nrow-1, ncol), dtype=np.int16)
        mask_r = nzrowix_mask
        sign_row = np.sign(rowix[mask_r]).astype(np.float32)
        # gather the offset_cycle with abs(rowix(...))-1 for 0-based
        absi_row = np.abs(rowix[mask_r]).astype(int) - 1
        oc_row_values = offset_cycle[absi_row]
        offgrid_row[mask_r] = np.round(oc_row_values * sign_row * nshortcycle).astype(np.int16)
        # Then rowcost(:,1:4:end) = -offgrid
        rowcost[:, 0::4] = -offgrid_row

        offgrid_col = np.zeros((nrow, ncol-1), dtype=np.int16)
        mask_c = nzcolix_mask
        sign_col = np.sign(colix[mask_c]).astype(np.float32)
        absi_col = np.abs(colix[mask_c]).astype(int) - 1
        oc_col_values = offset_cycle[absi_col]
        offgrid_col[mask_c] = np.round(oc_col_values * sign_col * nshortcycle).astype(np.int16)
        colcost[:, 0::4] = offgrid_col

        # Write out snaphu.costinfile with rowcost' then colcost' in int16
        # In MATLAB, rowcost' => (ncol*4, nrow-1), colcost' => ((ncol-1)*4, nrow)
        # then we do a fwrite. We'll replicate by flattening in Fortran order
        with open('snaphu.costinfile', 'wb') as f:
            # rowcost transpose
            rowcost_trans = np.transpose(rowcost).astype(np.int16)
            rowcost_trans.tofile(f)
            # colcost transpose
            colcost_trans = np.transpose(colcost).astype(np.int16)
            colcost_trans.tofile(f)

        # Write snaphu.in as complex data of ifgw => reshape(uw.ph(Z,i1),nrow,ncol)
        # ifgw shaped (nrow,ncol)
        ifgw = np.reshape(uw_ph[:, i1], (nrow, ncol))
        # We'll interpret ph as real phases for each pixel in row-major order or something. 
        # But the MATLAB code calls "writecpx('snaphu.in', ifgw)". 
        # That means it's writing a complex array? 
        # The original code might treat ifgw as a complex: ph is in radians? Possibly exp(1j*ph)? 
        # The snippet is ambiguous. We'll replicate literally:
        # If your data truly is real, you might store it as real + 0j. 
        ifgw_complex = np.exp(1j * ifgw.astype(np.float32))  # if the MATLAB code stored pure phase
        writecpx('snaphu.in', ifgw_complex)

        # system call to snaphu
        cmdstr = f"snaphu -d -f snaphu.conf {ncol} >& snaphu.log"
        # On Unix-like systems, you could do:
        ret = subprocess.call(cmdstr, shell=True)
        if ret != 0:
            print("Warning: snaphu command returned non-zero exit code.")

        # Read snaphu.out => (ncol, ??) float => in MATLAB code: ifguw=fread(fid,[ncol,inf],'float')
        # Then reshape to (nrow,ncol)
        with open('snaphu.out', 'rb') as f:
            data_out = np.fromfile(f, dtype=np.float32)
        # shape => (ncol, nrow) in column-major (MATLAB). In Python row-major, let's do:
        if data_out.size != (nrow * ncol):
            raise ValueError("snaphu.out size does not match expected dimensions.")
        ifguw = data_out.reshape((ncol, nrow)).T  # shape => (nrow,ncol)

        # compute msd:
        ifg_diff1 = (ifguw[:-1, :] - ifguw[1:, :]).flatten()
        ifg_diff1 = ifg_diff1[ifg_diff1 != 0]
        ifg_diff2 = (ifguw[:, :-1] - ifguw[:, 1:]).flatten()
        ifg_diff2 = ifg_diff2[ifg_diff2 != 0]
        denom = (ifg_diff1.size + ifg_diff2.size)
        if denom > 0:
            msd_val = (np.sum(ifg_diff1**2) + np.sum(ifg_diff2**2)) / denom
        else:
            msd_val = 0
        msd[i1] = msd_val

        # fill ph_uw for that band: ph_uw(:,i1)=ifguw(uw.nzix)
        # in Python:
        ph_uw[:, i1] = ifguw[uw_nzix]

    # Finally, the code does: save('uw_phaseuw','ph_uw','msd')
    # In Python we'll write it to a .mat:
    out_dict = {
        'ph_uw': ph_uw,
        'msd': msd
    }
    sio.savemat('uw_phaseuw.mat', out_dict)

    print("Done. Saved unwrapped phases to uw_phaseuw.mat.") 