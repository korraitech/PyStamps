import numpy as np
import os
from ..misc import get_module_info
from ..logger import appLogger
from .utils import read_h5,save_h5
from .uw_grid_wrapped import uw_grid_wrapped
from .uw_space_time import uw_space_time
from .uw_stat_costs import uw_stat_costs
from scipy.spatial import cKDTree

def uw_interp(workdir:str):
    print('Interpolating grid...')

    uw = read_h5(os.path.join(workdir,'uw_grid.h5'))
    y, x = np.nonzero(uw['nzix'])
    xy = np.column_stack((x, y))
    xy = xy[np.lexsort((xy[:,1], xy[:,0]))] + 1
    
    nrow, ncol = uw['nzix'].shape
    X, Y = np.meshgrid(np.arange(1 ,ncol+1), np.arange(1,nrow+1))
    
    query_points = np.column_stack((X.flatten(order='F'), Y.flatten(order='F')))
    _, Z = cKDTree(xy).query(query_points)

    Zvec_col  = Z.reshape(nrow,ncol).flatten(order='F')
    grid_edges = np.column_stack((Zvec_col[:-nrow],Zvec_col[nrow:]))
    grid_edges = np.vstack((grid_edges, np.column_stack((Z[:-ncol], Z[ncol:]))))

    sort_edges = np.sort(grid_edges, axis=1)
    I_sort = np.argsort(grid_edges, axis=1)
    edge_sign = I_sort[:, 1] - I_sort[:, 0]

    alledges, J = np.unique(sort_edges, axis=0, return_inverse=True)
    sameix = (alledges[:, 0] == alledges[:, 1])
    alledges[sameix, :] = 0
    
    edgs, J2 = np.unique(alledges, axis=0,return_inverse=True)
    n_edge = len(edgs) - 1
    edgs = np.column_stack((np.arange(n_edge), edgs[1:]))

    gridedgeix = (J2[J]) * edge_sign
    colix = gridedgeix[:nrow*(ncol-1)].reshape(nrow, ncol-1)
    rowix = gridedgeix[nrow*(ncol-1):].reshape(ncol, nrow-1).T

    print(f'   Number of unique edges in grid: {n_edge}')
    
    # Save the results
    save_h5(workdir,'uw_interp.h5',**{'edgs':edgs, 'n_edge':n_edge, 
                                      'rowix':rowix, 'colix':colix, 
                                      'Z':Z.reshape(nrow,ncol)})

def uw_unwrap_from_grid(workdir:str):
    print('Unwrapping from grid...')

    uw = read_h5(os.path.join(workdir,'uw_grid.h5'))
    uu = read_h5(os.path.join(workdir,'uw_phaseuw.h5'))

    n_ps, n_ifg = uw['ph_in'].shape
    gridix = np.zeros_like(uw['nzix'])
    gridix[uw['nzix']] = np.arange(uw['n_ps'])

    ph_uw = np.zeros((n_ps, n_ifg), dtype=np.float32)

    for i in range(n_ps):
        ix = gridix[uw['grid_ij'][i, 0], uw['grid_ij'][i, 1]]
        if ix == 0:
            ph_uw[i, :] = np.nan  # wrapped phase values were zero
        else:
            ph_uw_pix = uu['ph_uw'][ix, :]
            if np.isreal(uw['ph_in']):
                ph_uw[i, :] = ph_uw_pix + np.angle(np.exp(1j * (uw['ph_in'][i, :] - ph_uw_pix)))
            else:
                ph_uw[i, :] = ph_uw_pix + np.angle(uw['ph_in'][i, :] * np.exp(-1j * ph_uw_pix))
    
    msd = uu['msd']
    
    return ph_uw, msd 

def step_6_ps_unwrap(workdir:str,parms:dict):
    """
    Unwrap phase using the 3-D cost function phase unwrapping algorithm.
    
    Args:
        workdir:str - path to the working directory
        parms:dict - parameters from parms.json
    """
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {}".format(
            get_module_info(),workdir)
    )

    # Load version info
    psver = int(read_h5(os.path.join(workdir,'psver.h5'))['psver'])
    
    # Define filenames
    psname = f'ps{psver}.h5'
    rcname = f'rc{psver}.h5'
    pmname = f'pm{psver}.h5'
    bpname = f'bp{psver}.h5'
    phuwname = f'phuw{psver}.h5'

    ps = read_h5(os.path.join(workdir,psname))
    n_ifg = ps['n_ifg']
    master_ix = ps['master_ix']
    n_ps = ps['n_ps']
    unwrap_ifg_index = np.setdiff1d(np.arange(ps['n_ifg']), [])

    if os.path.exists(os.path.join(workdir,bpname)):
        bp = read_h5(os.path.join(workdir,bpname))
    else:
        bperp = ps['bperp']
        bperp = np.concatenate((bperp[:master_ix], bperp[master_ix+1:]))
        bp['bperp_mat'] = np.tile(bperp, (n_ps, 1))

    # Create bperp_mat
    bperp_mat = np.hstack([
        bp['bperp_mat'][:, :master_ix],
        np.zeros((n_ps, 1), dtype=np.float32),
        bp['bperp_mat'][:, master_ix:]
    ])

    # Handle phase data
    rc = read_h5(os.path.join(workdir,rcname))
    ph_w = rc['ph_rc']
    if os.path.exists(os.path.join(workdir,pmname)):
        pm = read_h5(os.path.join(workdir,pmname))
        if 'K_ps' in pm and pm['K_ps'] is not None:
            ph_w *= np.exp(1j * (np.tile(pm['K_ps'], (1, n_ifg)) * bperp_mat))

    # Normalize phase values
    ix = ph_w != 0
    ph_w[ix] = ph_w[ix] / np.abs(ph_w[ix])  # normalize to avoid high freq artifacts

    max_topo_err = int(parms['max_topo_err'])
    wavelength = float(parms['lambda'])

    # ===============================================
    # The code below needs to be made sensor specific
    # ===============================================
    rho = 830000  # mean range - need only be approximately correct
    inc_mean = ps['mean_incidence']
    max_K = max_topo_err / (wavelength * rho * np.sin(inc_mean) / 4 / np.pi)
    # ===============================================
    # The code above needs to be made sensor specific
    # ===============================================

    # Calculate number of trial wraps
    bperp_range = np.max(ps['bperp']) - np.min(ps['bperp'])
    
    # Set unwrapping options
    options = {'master_day': ps['master_day']}
    options.update({
        'time_win': int(parms['unwrap_time_win']),
        'unwrap_method': parms['unwrap_method'],
        'grid_size': int(parms['unwrap_grid_size']),
        'prefilt_win': int(parms['unwrap_gold_n_win']),
        'goldfilt_flag': parms['unwrap_prefilter_flag'],
        'gold_alpha': float(parms['unwrap_gold_alpha']),
        'la_flag': parms['unwrap_la_error_flag'],
        'scf_flag': parms['unwrap_spatial_cost_func_flag']
    })

    options.setdefault('lowfilt_flag', 'n')
    options.setdefault('temp', None)
    options.setdefault('n_temp_wraps', 2)
    options.setdefault('max_bperp_for_temp_est', 100)
    options.setdefault('variance', None)
    options.setdefault('unwrap_method', '3D_FULL')
    
    options['n_trial_wraps'] = (bperp_range * max_K / (2 * np.pi))
    print(f'n_trial_wraps={options["n_trial_wraps"]}')

    # Handle day indices
    ifgday_ix = np.column_stack([np.ones(n_ifg) * master_ix,np.arange(0, n_ifg)])
    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, master_ix)
    day = ps['day'] - ps['master_day']

    # Calling unwrapping function Snaphu
    uw_grid_wrapped(workdir,ph_w[:, unwrap_ifg_index],ps['xy'],options)
    uw_interp(workdir)
    uw_space_time(workdir,day, ifgday_ix[unwrap_ifg_index], 
                  ps['bperp'][unwrap_ifg_index],options['time_win'],  options['n_trial_wraps'])
    uw_stat_costs(workdir)
    ph_uw_some, msd_some = uw_unwrap_from_grid(workdir)
    
    # Initialize output arrays
    ph_uw = np.zeros((n_ps, n_ifg), dtype='float32')
    msd = np.zeros(n_ifg, dtype='float32')
    
    # Assign unwrapped results
    ph_uw[:, unwrap_ifg_index] = ph_uw_some
    msd[unwrap_ifg_index] = msd_some

    # Zero out non-unwrapped interferograms
    non_unwrap_ifgs = np.setdiff1d(np.arange(n_ifg), unwrap_ifg_index)
    ph_uw[:, non_unwrap_ifgs] = 0

    # Save results
    save_h5(workdir,phuwname, **{'ph_uw': ph_uw, 'msd': msd})
    