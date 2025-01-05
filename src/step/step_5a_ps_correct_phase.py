import numpy as np
import os
from .utils import read_h5,save_h5

def step_5a_ps_correct_phase(workdir:str, patch:str):
    # PS_CORRECT_PHASE() correct phase from estimate of look angle error
    #
    #   Andy Hooper, June 2006
    #
    #   ==========================================================
    #   07/2006 AH: Use specific bperp for correction
    #   09/2006 AH: add small baselines 
    #   ==========================================================

    print("Running Step-07 ...\t[{}]".format(patch))
    print('Correcting phase for look angle error...')

    patch_dir = os.path.join(workdir, patch)

    psver = int(read_h5(os.path.join(patch_dir, 'psver.h5'))['psver'])
    psname = f'ps{psver}.h5'
    phname = f'ph{psver}.h5'
    pmname = f'pm{psver}.h5'
    bpname = f'bp{psver}.h5'

    pm = read_h5(os.path.join(patch_dir, pmname))
    K_ps = pm['K_ps'].astype(np.float32)
    C_ps = pm['C_ps'].astype(np.float32)

    ps = read_h5(os.path.join(patch_dir, psname))
    master_ix = np.sum(ps['master_day'] > ps['day']) + 1

    bp = read_h5(os.path.join(patch_dir, bpname))
    bperp_mat = np.hstack(
        [bp['bperp_mat'][:, :ps['master_ix'] - 1],
         np.zeros((ps['n_ps'], 1), dtype=np.float32),
         bp['bperp_mat'][:, ps['master_ix']:]]
    )

    ph = read_h5(os.path.join(patch_dir, phname))['ph']
    ph_rc = ph * np.exp(-1j * (np.tile(K_ps, (1, ps['n_ifg'])) * bperp_mat + np.tile(C_ps, (1, ps['n_ifg']))))
    ph_reref = np.hstack([
        pm['ph_patch'][:, :master_ix - 1].astype(np.float32),
        np.ones((ps['n_ps'], 1), dtype=np.float32),
        pm['ph_patch'][:, master_ix:].astype(np.float32)
    ])

    rcname = f'rc{psver}.h5'
    save_h5(patch_dir, rcname, **{'ph_rc': ph_rc, 'ph_reref': ph_reref})
