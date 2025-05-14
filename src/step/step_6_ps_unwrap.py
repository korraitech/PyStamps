import numpy as np
import os
from ..misc import get_module_info
from ..logger import appLogger
from .utils import read_h5,save_h5
from .uw_grid_wrapped import uw_grid_wrapped

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
    # ph_uw_some, msd_some = uw_3d( ph_w[:, unwrap_ifg_index], ps['xy'], day, 
    #     ifgday_ix[unwrap_ifg_index], ps['bperp'][unwrap_ifg_index], options
    # )

    # # Initialize output arrays
    # ph_uw = np.zeros((n_ps, n_ifg), dtype='float32')
    # msd = np.zeros(n_ifg, dtype='float32')
    
    # # Assign unwrapped results
    # ph_uw[:, unwrap_ifg_index] = ph_uw_some
    # if 'msd_some' in locals():
    #     msd[unwrap_ifg_index] = msd_some

    # # Zero out non-unwrapped interferograms
    # non_unwrap_ifgs = np.setdiff1d(np.arange(n_ifg), unwrap_ifg_index)
    # ph_uw[:, non_unwrap_ifgs] = 0

    # # Save results
    # save_h5(workdir,phuwname, **{'ph_uw': ph_uw, 'msd': msd})
    