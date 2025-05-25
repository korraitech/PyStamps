import numpy as np
import os
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from datetime import date
from ..misc import get_module_info
from ..logger import appLogger
from .utils import read_h5, save_h5,lscov
from .ps_deramp import ps_deramp
from .ps_setref import ps_setref

def smooth_scla(workdir:str):
    print('Smoothing spatially-correlated look angle error...')

    psver = int(read_h5(os.path.join(workdir, 'psver.h5'))['psver'])

    psname = f'ps{psver}.h5'
    bpname = f'bp{psver}.h5'
    sclaname = f'scla{psver}.h5'
    sclasmoothname = f'scla_smooth{psver}.h5'

    ps = read_h5(os.path.join(workdir, psname))
    scla = read_h5(os.path.join(workdir, sclaname))
    K_ps_uw = scla['K_ps_uw']
    C_ps_uw = scla['C_ps_uw']
    ph_ramp = scla['ph_ramp']

    n_ps = int(ps['n_ps'])
    print(f'Number of points per ifg: {n_ps}')

    tri = Delaunay(ps['xy'][:, 1:])
    simplices = tri.simplices
    edges = np.vstack((simplices[:, [0, 1]],simplices[:, [1, 2]],simplices[:, [0, 2]]))
    edgs = np.unique(np.sort(edges, axis=1), axis=0)
    n_edge = edgs.shape[0]

    print(f'Number of arcs per ifg={n_edge}')

    Kneigh_min = np.full(n_ps, np.inf, dtype=np.float32)
    Kneigh_max = np.full(n_ps, -np.inf, dtype=np.float32)
    Cneigh_min = np.full(n_ps, np.inf, dtype=np.float32)
    Cneigh_max = np.full(n_ps, -np.inf, dtype=np.float32)

    for i in range(n_edge):
        ix = edgs[i]
        Kneigh_min[ix] = np.minimum(Kneigh_min[ix], K_ps_uw[ix[::-1]])
        Kneigh_max[ix] = np.maximum(Kneigh_max[ix], K_ps_uw[ix[::-1]])
        Cneigh_min[ix] = np.minimum(Cneigh_min[ix], C_ps_uw[ix[::-1]])
        Cneigh_max[ix] = np.maximum(Cneigh_max[ix], C_ps_uw[ix[::-1]])
        if i % 100000 == 0:
            print(f'{i} arcs processed', 2)

    ix1 = K_ps_uw > Kneigh_max
    ix2 = K_ps_uw < Kneigh_min
    K_ps_uw[ix1] = Kneigh_max[ix1]
    K_ps_uw[ix2] = Kneigh_min[ix2]

    ix1 = C_ps_uw > Cneigh_max
    ix2 = C_ps_uw < Cneigh_min
    C_ps_uw[ix1] = Cneigh_max[ix1]
    C_ps_uw[ix2] = Cneigh_min[ix2]

    bp = read_h5(os.path.join(workdir, bpname))
    bperp_mat = np.insert(bp['bperp_mat'], ps['master_ix'] - 1, 0, axis=1)
    ph_scla = np.outer(K_ps_uw, np.ones(bperp_mat.shape[1])) * bperp_mat

    save_h5(workdir, sclasmoothname, **{'K_ps_uw': K_ps_uw, 
        'C_ps_uw': C_ps_uw, 'ph_scla': ph_scla, 'ph_ramp': ph_ramp})

def step_7_ps_scla(workdir:str, parms:dict):
    """
    Estimate spatially-correlated look angle error.
    """
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {}".format(
            get_module_info(),workdir)
    )

    scla_method = parms['scla_method']
    scla_deramp = parms['scla_deramp']

    psver = int(read_h5(os.path.join(workdir, 'psver.h5'))['psver'])
    psname = f'ps{psver}.h5'
    bpname = f'bp{psver}.h5'
    ifgstdname = f'ifgstd{psver}.h5'
    phuwname = f'phuw{psver}.h5'
    sclaname = f'scla{psver}.h5'
        
    ps = read_h5(os.path.join(workdir, psname))
    bp = read_h5(os.path.join(workdir, bpname))
    uw = read_h5(os.path.join(workdir, phuwname))

    unwrap_ifg_index = np.arange(ps['n_ifg'])

    ph_ramp = np.array([])
    if scla_deramp.lower() == 'y':
        print('\n   deramping ifgs...\n')
        _, ph_ramp = ps_deramp(ps, uw['ph_uw'].copy())
        uw['ph_uw'] = uw['ph_uw'] - ph_ramp

    ref_ps = ps_setref(workdir, parms)
    uw['ph_uw'] = uw['ph_uw'] - np.nanmean(uw['ph_uw'][ref_ps, :], axis=0)

    master_ix = int(ps['master_ix'])
    n_ps = int(ps['n_ps'])
    bperp_mat = np.hstack((bp['bperp_mat'][:, :master_ix], np.zeros((n_ps, 1), dtype='float32'), bp['bperp_mat'][:, master_ix:]))
    day = np.diff(ps['day'][unwrap_ifg_index])
    ph = np.diff(uw['ph_uw'][:, unwrap_ifg_index], axis=1)
    bperp = np.diff(bperp_mat[:, unwrap_ifg_index], axis=1)

    bprint = np.mean(bperp, axis=0)
    print(f'{ph.shape[1]} ifgs used in estimation:')
    for i in range(ph.shape[1]):
        start_day = date.fromordinal(int(ps["day"][unwrap_ifg_index[i]]))
        end_day = date.fromordinal(int(ps["day"][unwrap_ifg_index[i] + 1]))
        print(f'{start_day.strftime("%d-%b-%Y")} to {end_day.strftime("%d-%b-%Y")} {day[i]:4d} days {bprint[i]:4.0f} m')

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
    w = np.diag(np.linalg.inv(ifg_vcm[np.ix_(unwrap_ifg_index, unwrap_ifg_index)]))
    m = lscov(G, (uw['ph_uw'][:, unwrap_ifg_index] - ph_scla[:, unwrap_ifg_index]).T, w)
    C_ps_uw = m[0, :]

    save_h5(workdir, sclaname, **{'ph_scla': ph_scla, 'K_ps_uw': K_ps_uw, 
                                  'C_ps_uw': C_ps_uw, 'ph_ramp': ph_ramp, 
                                  'ifg_vcm': ifg_vcm})
    smooth_scla(workdir)
