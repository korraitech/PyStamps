import numpy as np
from .utils import read_h5,save_h5
import os

def run_snaphu(workdir,rowcost,colcost,ifgw,ncol):
    # Write cost file
    with open(os.path.join(workdir,'snaphu.costinfile'), 'wb') as fid:
        fid.write(rowcost.astype(np.int16).tobytes())
        fid.write(colcost.astype(np.int16).tobytes())

    # Write input file
    vname_flt = np.zeros((ifgw.shape[0], ifgw.shape[1] * 2), dtype='=f4')
    vname_flt[:, 0::2] = np.real(ifgw)
    vname_flt[:, 1::2] = np.imag(ifgw)
    with open(os.path.join(workdir,'snaphu.in'), 'wb') as fid:
        vname_flt.T.astype('=f4').tobytes('F')
        fid.write(vname_flt.T.tobytes('F'))

    # Run snaphu
    cmdstr = f'snaphu -d -f {os.path.join(workdir,"snaphu.conf")} {ncol} > {os.path.join(workdir,"snaphu.log")}'
    os.system(cmdstr)

    # Read output
    with open(os.path.join(workdir,'snaphu.out'), 'rb') as fid:
        # ifguw = np.fromfile(fid, dtype=np.float32).reshape(ncol, -1).T
        ifguw = np.fromfile(fid, dtype='=f4').reshape(-1, ncol).T
    return ifguw

def uw_stat_costs(workdir):
    costscale = 100
    nshortcycle = 200
    maxshort = 32000

    print('Unwrapping in space...')

    uw = read_h5(os.path.join(workdir,'uw_grid.h5'))
    ui = read_h5(os.path.join(workdir,'uw_interp.h5'))
    ut = read_h5(os.path.join(workdir,'uw_space_time.h5'))

    ph = uw['ph']
    nzix = uw['nzix']
    n_ps = uw['n_ps']
    n_ifg = uw['n_ifg']
    dph_space_uw = ut['dph_space_uw']
    dph_noise = ut['dph_noise']

    subset_ifg_index = np.arange(ph.shape[1])
    nrow, ncol = nzix.shape

    colix = ui['colix']
    rowix = ui['rowix']
    Z = np.ravel(ui['Z'], order='F')

    grid_edges = np.abs(np.concatenate((colix[np.abs(colix) > 0], rowix[np.abs(rowix) > 0])))
    bins = np.arange(1, ui['n_edge'] + 1)
    n_edges = np.histogram(grid_edges, bins=np.append(bins, bins[-1] + 1))[0]

    sigsq_noise = (np.std(dph_noise, axis=1, ddof=1) / (2 * np.pi))**2
    dph_smooth = dph_space_uw - dph_noise

    nostats_ix = np.where(np.isnan(sigsq_noise))[0]

    rowix = rowix.astype(np.float32)
    colix = colix.astype(np.float32)
    for i in nostats_ix:
        idx = i +1
        rowix[np.abs(rowix) == idx] = np.nan
        colix[np.abs(colix) == idx] = np.nan

    sigsq = np.round((sigsq_noise * nshortcycle * nshortcycle) / costscale* n_edges)
    sigsq = np.int16(np.nan_to_num(sigsq, nan=0))
    sigsq[sigsq < 1] = 1

    rowcost = np.zeros((nrow-1, ncol*4), dtype=np.int16)
    colcost = np.zeros((nrow, (ncol-1)*4), dtype=np.int16)
    nzrowix = np.abs(rowix) > 0
    nzcolix = np.abs(colix) > 0
    rowstdgrid = np.ones(rowix.shape, dtype=np.int16)
    colstdgrid = np.ones(colix.shape, dtype=np.int16)
    rowcost[:, 2::4] = maxshort
    colcost[:, 2::4] = maxshort
    stats_ix = ~np.isnan(rowix)
    rowcost[:, 3::4] = np.int16(stats_ix) * (-1 - maxshort) + 1
    stats_ix = ~np.isnan(colix)
    colcost[:, 3::4] = np.int16(stats_ix) * (-1 - maxshort) + 1

    ph_uw = np.zeros((n_ps, n_ifg), dtype=np.float32)
    ifguw = np.zeros((nrow, ncol))
    msd = np.zeros(n_ifg)

    # Write snaphu configuration
    with open(os.path.join(workdir,'snaphu.conf'), 'w') as fid:
        fid.write(f'INFILE  {os.path.join(workdir,"snaphu.in")}\n')
        fid.write(f'OUTFILE {os.path.join(workdir,"snaphu.out")}\n')
        fid.write(f'COSTINFILE {os.path.join(workdir,"snaphu.costinfile")}\n')
        fid.write('STATCOSTMODE  DEFO\n')
        fid.write('INFILEFORMAT  COMPLEX_DATA\n')
        fid.write('OUTFILEFORMAT FLOAT_DATA\n')

    # Process each interferogram
    for i1 in subset_ifg_index:
        print(f'   Processing IFG {i1+1} of {len(subset_ifg_index)}')

        rowidx_flat = rowix.flatten('F')[nzrowix.flatten('F')]
        colidx_flat = colix.flatten('F')[nzcolix.flatten('F')]

        rowstdgrid[nzrowix] = sigsq[np.abs(rowidx_flat).astype(np.int16)-1]
        rowcost[:, 1::4] = rowstdgrid
        colstdgrid[nzcolix] = sigsq[np.abs(colidx_flat).astype(np.int16)-1]
        colcost[:, 1::4] = colstdgrid

        offset_cycle = (np.angle(np.exp(1j * dph_space_uw[:, i1])) - dph_smooth[:, i1]) / (2 * np.pi)

        offgrid = np.zeros(rowix.shape, dtype=np.int16)
        offgrid[nzrowix] = np.round(offset_cycle[np.abs(rowidx_flat).astype(np.int16) -1] * np.sign(rowidx_flat) * nshortcycle).astype(np.int16)
        rowcost[:, 0::4] = -offgrid

        offgrid = np.zeros(colix.shape, dtype=np.int16)
        offgrid[nzcolix] = np.round(offset_cycle[np.abs(colidx_flat).astype(np.int16) -1] * np.sign(colidx_flat) * nshortcycle).astype(np.int16)
        colcost[:, 0::4] = offgrid

        ifgwght = ph[Z - 1, i1].reshape(nrow, ncol, order='F')

        ifguw = run_snaphu(workdir,rowcost,colcost,ifgwght,ncol)

        ifg_diff1 = ifguw[:-1, :] - ifguw[1:, :]
        ifg_diff1 = ifg_diff1[ifg_diff1 != 0]
        ifg_diff2 = ifguw[:, :-1] - ifguw[:, 1:]
        ifg_diff2 = ifg_diff2[ifg_diff2 != 0]
        msd[i1] = (np.sum(ifg_diff1**2) + np.sum(ifg_diff2**2)) / (len(ifg_diff1) + len(ifg_diff2))

        ph_uw[:, i1] = ifguw.flatten(order='F')[nzix.flatten(order='F')]

    # Save results
    save_h5(workdir,"uw_phaseuw.h5",**{
        'ph_uw': ph_uw,
        'msd': msd
    })
