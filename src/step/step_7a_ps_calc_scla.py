import numpy as np
import os
from scipy.optimize import minimize
from datetime import datetime
from .utils import read_h5, save_h5
from .ps_deramp import ps_deramp
from .ps_setref import ps_setref

def step_7a_ps_calc_scla(workdir:str, parms:dict):

    print("Running Step-07a ...")
    print('Estimating spatially-correlated look angle error...')

    drop_ifg_index = parms['drop_ifg_index']
    scla_method = parms['scla_method']
    scla_deramp = parms['scla_deramp']
    scla_drop_index = parms['scla_drop_index']

    psver = int(read_h5(os.path.join(workdir, 'psver.h5'))['psver'])
    psname = f'ps{psver}.h5'
    bpname = f'bp{psver}.h5'
    ifgstdname = f'ifgstd{psver}.h5'
    phuwname = f'phuw{psver}.h5'
    sclaname = f'scla{psver}.h5'
        
    ps = read_h5(os.path.join(workdir, psname))
    bp = read_h5(os.path.join(workdir, bpname))
    uw = read_h5(os.path.join(workdir, phuwname))

    unwrap_ifg_index = np.setdiff1d(np.arange(1, ps['n_ifg'] + 1), drop_ifg_index)

    if scla_deramp.lower() == 'y':
        print('\n   deramping ifgs...\n')
        _, ph_ramp = ps_deramp(ps, uw['ph_uw'])
        uw['ph_uw'] = uw['ph_uw'] - ph_ramp
    else:
        ph_ramp = np.array([])

    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, scla_drop_index)

    ref_ps = ps_setref(workdir, parms)
    uw['ph_uw'] = uw['ph_uw'] - np.nanmean(uw['ph_uw'][ref_ps, :], axis=0)
    master_ix = int(ps['master_ix'])
    n_ps = int(ps['n_ps'])
    bperp_mat = np.hstack((bp['bperp_mat'][:, :master_ix - 1], np.zeros((n_ps, 1), dtype='float32'), bp['bperp_mat'][:, master_ix:]))
    day = np.diff(ps['day'][unwrap_ifg_index])
    ph = np.diff(uw['ph_uw'][:, unwrap_ifg_index], axis=1)
    bperp = np.diff(bperp_mat[:, unwrap_ifg_index], axis=1)

    bprint = np.mean(bperp, axis=0)
    print(f'{ph.shape[1]} ifgs used in estimation:')

    for i in range(ph.shape[1]):
        print(f'   {datetime.fromordinal(int(ps["day"][unwrap_ifg_index[i]]))} to {datetime.fromordinal(int(ps["day"][unwrap_ifg_index[i] + 1]))} {day[i]} days {round(bprint[i])} m')

    K_ps_uw = np.zeros(n_ps)

    if len(unwrap_ifg_index) < 4:
        G = np.vstack((np.ones(ph.shape[1]), np.mean(bperp, axis=0))).T
    else:
        G = np.vstack((np.ones(ph.shape[1]), np.mean(bperp, axis=0), day)).T

    ifgstd = read_h5(os.path.join(workdir, ifgstdname))
    ifg_vcm = np.diag((ifgstd['ifg_std'] * np.pi / 180) ** 2)

    m = np.linalg.lstsq(G, ph.T, rcond=None)[0]
    K_ps_uw = m[1, :]

    if scla_method == 'L1':
        for i in range(n_ps):
            d = ph[i, :]
            m2 = m[:, i]
            res = minimize(lambda x: np.sum(np.abs(d - G @ x)), m2)
            K_ps_uw[i] = res.x[1]

    ph_scla = np.outer(K_ps_uw, np.ones(bperp_mat.shape[1])) * bperp_mat
    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, master_ix)
    G = np.vstack((np.ones(len(unwrap_ifg_index)), ps['day'][unwrap_ifg_index] - ps['day'][master_ix])).T
    m = np.linalg.lstsq(G, (uw['ph_uw'][:, unwrap_ifg_index] - ph_scla[:, unwrap_ifg_index]).T, rcond=None)[0]
    C_ps_uw = m[0, :]

    # oldscla = os.path.exists(f'{sclaname}.mat')
    # if oldscla:
    #     olddatenum = datetime.fromtimestamp(os.path.getmtime(f'{sclaname}.mat'))
    #     os.rename(f'{sclaname}.mat', f'tmp_{sclaname[3:]}{olddatenum.strftime("_%Y%m%d_%H%M%S")}.mat')

    save_h5(workdir, sclaname, **{'ph_scla': ph_scla, 'K_ps_uw': K_ps_uw, 
                                  'C_ps_uw': C_ps_uw, 'ph_ramp': ph_ramp, 
                                  'ifg_vcm': ifg_vcm})
