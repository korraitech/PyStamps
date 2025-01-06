import numpy as np
import os
import platform
from .utils import read_h5,save_h5
from .ps_plot_tca import ps_plot_tca
from .uw_3d import uw_3d

def step_6_ps_unwrap(workdir:str,parms:dict):
    """
    Unwrap phase using the 3-D cost function phase unwrapping algorithm.
    
    Original MATLAB by Andy Hooper, Jun 2006
    Python translation includes core functionality
    """

    # Get parameters (you'll need to implement getparm() equivalent)
    unwrap_patch_phase = parms['unwrap_patch_phase']
    scla_deramp = parms['scla_deramp']
    subtr_tropo = parms['subtr_tropo']
    aps_name = parms['tropo_method']
    drop_ifg_index = parms['drop_ifg_index']

    # Load version info
    psver = int(read_h5(os.path.join(workdir,'psver.h5'))['psver'])
    
    # Define filenames
    psname = f'ps{psver}.h5'
    rcname = f'rc{psver}.h5'
    pmname = f'pm{psver}.h5'
    bpname = f'bp{psver}.h5'
    sclaname = f'scla_smooth{psver}.h5'
    apsname = f'tca{psver}.h5'
    phuwname = f'phuw{psver}.h5'

    # Load PS data
    ps = read_h5(os.path.join(workdir,psname))

    # Get interferogram indices
    unwrap_ifg_index = np.setdiff1d(
        np.arange(ps['n_ifg'][0][0]), drop_ifg_index)

    # Load or create bperp data
    bp = read_h5(os.path.join(workdir,bpname))

    # Create bperp_mat
    master_ix = ps['master_ix'][0][0]
    zeros_col = np.zeros((ps['n_ps'][0][0], 1), dtype='float32')
    bperp_mat = np.hstack([
        bp['bperp_mat'][:, :master_ix-1],
        zeros_col,
        bp['bperp_mat'][:, master_ix-1:]
    ])

    # Handle phase data
    if str(unwrap_patch_phase).lower() == 'y':
        pm = read_h5(os.path.join(workdir,pmname))
        ph_w = pm['ph_patch'] / np.abs(pm['ph_patch'])
        ones_col = np.ones((ps['n_ps'], 1))
        ph_w = np.hstack([ph_w[:, :ps['master_ix']-1], 
                        ones_col, 
                        ph_w[:, ps['master_ix']:]])
    else:
        rc = read_h5(os.path.join(workdir,rcname))
        ph_w = rc['ph_rc']
        if os.path.exists(f'./{pmname}.mat'):
            pm = read_h5(os.path.join(workdir,pmname))
            if 'K_ps' in pm and pm['K_ps'] is not None:
                ph_w *= np.exp(1j * (np.tile(pm['K_ps'], (1, ps['n_ifg'])) * bperp_mat))

    # Normalize phase values
    ix = ph_w != 0
    ph_w[ix] = ph_w[ix] / np.abs(ph_w[ix])  # normalize to avoid high freq artifacts

    scla_subtracted_sw = False
    ramp_subtracted_sw = False

    # Set up options
    options = {'master_day': ps['master_day'][0][0]}

    # Handle PS case
    print('   subtracting scla and master aoe...')
    scla = read_h5(os.path.join(workdir,sclaname))
    
    if scla['K_ps_uw'].shape[0] == ps['n_ps'][0][0]:
        scla_subtracted_sw = True
        # Subtract spatially correlated look angle error
        ph_w = ph_w * np.exp(-1j * np.tile(scla['K_ps_uw'], (1, ps['n_ifg'][0][0])) * bperp_mat)
        # Subtract master APS
        ph_w = ph_w * np.tile(np.exp(-1j * scla['C_ps_uw']), (1, ps['n_ifg'][0][0]))
        
        if (scla_deramp.lower() == 'y' and 'ph_ramp' in scla and 
            scla['ph_ramp'].shape[0] == ps['n_ps'][0][0]):
            ramp_subtracted_sw = True
            ph_w = ph_w * np.exp(-1j * scla['ph_ramp'])  # subtract orbital ramps
    else:
        print('   wrong number of PS in scla - subtraction skipped...')
        if os.path.exists(f'{sclaname}.mat'):
            os.remove(f'{sclaname}.mat')

    # Handle APS (atmospheric phase screen) correction
    if os.path.exists(f'{apsname}.mat') and subtr_tropo.lower() == 'y':
        print('   subtracting slave aps...')
        aps = read_h5(os.path.join(workdir,apsname))
        aps_corr, fig_name_tca, aps_flag = ps_plot_tca(aps, aps_name)
        ph_w = ph_w * np.exp(-1j * aps_corr)
        del aps

    # Set unwrapping options
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

    # Get additional parameters
    max_topo_err = int(parms['max_topo_err'])
    wavelength = float(parms['lambda'])

    # Sensor specific calculations
    rho = 830000  # mean range - need only be approximately correct
    if 'mean_incidence' in ps:
        inc_mean = ps['mean_incidence']
    else:
        laname = f'./la{psver}.mat'
        if os.path.exists(laname):
            la = read_h5(os.path.join(workdir,laname))
            inc_mean = np.mean(la['la']) + 0.052  # incidence angle ≈ look angle + 3 deg
        else:
            inc_mean = 21 * np.pi / 180  # guess the incidence angle

    max_K = max_topo_err / (wavelength * rho * np.sin(inc_mean) / 4 / np.pi)

    # Calculate number of trial wraps
    bperp_range = np.max(ps['bperp']) - np.min(ps['bperp'])
    options['n_trial_wraps'] = (bperp_range * max_K / (2 * np.pi))
    print(f'n_trial_wraps={options["n_trial_wraps"]}')

    # Handle day indices
    ifgday_ix = np.column_stack([
        np.ones(ps['n_ifg'][0][0]) * ps['master_ix'][0][0],
        np.arange(1, ps['n_ifg'][0][0] + 1)
    ])
    master_ix = np.sum(ps['master_day'] > ps['day']) + 1
    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, master_ix)
    day = ps['day'] - ps['master_day']

    # Call appropriate unwrapping function based on platform
    # Rahul Sharan
    if not platform.system().lower().startswith('win'):
        ph_uw_some, msd_some = uw_3d(
            ph_w[:, unwrap_ifg_index], 
            ps['xy'], 
            day, 
            ifgday_ix[unwrap_ifg_index], 
            ps['bperp'][unwrap_ifg_index], 
            options
        )
    else:
        print('Windows detected: using old unwrapping code without statistical cost processing')
        ph_uw_some = uw_nosnaphu(ph_w[:, unwrap_ifg_index], ps['xy'], day, options)

    # Initialize output arrays
    ph_uw = np.zeros((ps['n_ps'][0][0], ps['n_ifg'][0][0]), dtype='float32')
    msd = np.zeros(ps['n_ifg'][0][0], dtype='float32')
    
    # Assign unwrapped results
    ph_uw[:, unwrap_ifg_index] = ph_uw_some
    if 'msd_some' in locals():
        msd[unwrap_ifg_index] = msd_some

    # Add back SCLA and master AOE for PS case
    if scla_subtracted_sw :
        print('Adding back SCLA and master AOE...')
        scla = read_h5(os.path.join(workdir,sclaname))
        # Add back spatially correlated look angle error
        ph_uw = ph_uw + (np.tile(scla['K_ps_uw'], (1, ps['n_ifg'][0][0])) * bperp_mat)
        # Add back master APS
        ph_uw = ph_uw + np.tile(scla['C_ps_uw'], (1, ps['n_ifg'][0][0]))
        if ramp_subtracted_sw:
            ph_uw = ph_uw + scla['ph_ramp']  # Add back orbital ramps

    # Add back slave APS if applicable
    if os.path.exists(f'{apsname}.mat') and subtr_tropo.lower() == 'y':
        print('Adding back slave APS...')
        aps = read_h5(os.path.join(workdir,apsname))
        aps_corr, fig_name_tca, aps_flag = ps_plot_tca(aps, aps_name)
        ph_uw = ph_uw + aps_corr

    # Handle patch phase unwrapping if enabled
    if unwrap_patch_phase.lower() == 'y':
        pm = read_h5(os.path.join(workdir,pmname))
        ph_w = pm['ph_patch'] / np.abs(pm['ph_patch'])
        ph_w = np.hstack([
            ph_w[:, :ps['master_ix'][0][0]-1],
            np.zeros((ps['n_ps'][0][0], 1)),
            ph_w[:, ps['master_ix'][0][0]-1:]
        ])
        rc = read_h5(os.path.join(workdir,rcname))
        ph_uw = ph_uw + np.angle(rc['ph_rc'] * np.conj(ph_w))

    # Zero out non-unwrapped interferograms
    non_unwrap_ifgs = np.setdiff1d(np.arange(ps['n_ifg'][0][0]), unwrap_ifg_index)
    ph_uw[:, non_unwrap_ifgs] = 0

    # Save results
    save_h5(workdir,phuwname, **{'ph_uw': ph_uw, 'msd': msd})
    