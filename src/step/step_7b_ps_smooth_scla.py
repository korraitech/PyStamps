import numpy as np
import os
from scipy.spatial import Delaunay
from .utils import read_h5, save_h5

def step_7b_ps_smooth_scla(workdir:str):

    print("Running Step-07b ...")
    print('Smoothing spatially-correlated look angle error...')

    psver = int(read_h5(os.path.join(workdir, 'psver.h5'))['psver'])

    psname = f'ps{psver}.h5'
    bpname = f'bp{psver}.h5'
    sclaname = f'scla{psver}.h5'

    ps = read_h5(os.path.join(workdir, psname))
    scla = read_h5(os.path.join(workdir, sclaname))
    K_ps_uw = scla['K_ps_uw']
    C_ps_uw = scla['C_ps_uw']
    ph_ramp = scla['ph_ramp']

    xy = ps['xy'].astype(float)
    tri = Delaunay(xy[:, 1:3])
    edgs = tri.convex_hull
    n_edge = edgs.shape[0]

    n_ps = int(ps['n_ps'])
    print(f'Number of arcs per ifg={n_edge}')
    print(f'Number of points per ifg: {n_ps}')

    Kneigh_min = np.full(n_ps, np.inf, dtype=np.float32)
    Kneigh_max = np.full(n_ps, -np.inf, dtype=np.float32)
    Cneigh_min = np.full(n_ps, np.inf, dtype=np.float32)
    Cneigh_max = np.full(n_ps, -np.inf, dtype=np.float32)

    for i in range(n_edge):
        ix = edgs[i, :1]
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

    sclasmoothname = f'scla_smooth{psver}.h5'
    save_h5(os.path.join(workdir, sclasmoothname), **{'K_ps_uw': K_ps_uw, 
        'C_ps_uw': C_ps_uw, 'ph_scla': ph_scla, 'ph_ramp': ph_ramp})
