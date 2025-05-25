import numpy as np
import os
from ..misc import get_module_info
from ..logger import appLogger
from .utils import read_h5, save_h5

def step_aps_linear(workdir: str):
    """
    Compute tropospheric delay map from linear relation between phase and topography.
    
    Args:
        workdir:str - path to the working directory
    """
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {}".format(
            get_module_info(),workdir)
    )

    psver = int(read_h5(os.path.join(workdir,'psver.h5'))['psver'])

    phuwname = f'phuw{psver}.h5'
    hgtname = f'hgt{psver}.h5'
    apsname = f'tca{psver}.h5'

    phuw = read_h5(os.path.join(workdir, phuwname))['ph_uw']
    n_dates = phuw.shape[1]

    hgt = read_h5(os.path.join(workdir, hgtname))['hgt']
    ix_points = np.arange(hgt.shape[0])

    ph_tropo_linear = np.zeros((hgt.shape[0], n_dates))
    hgt_range = np.array([np.min(hgt), np.max(hgt)])
    if hgt_range[1] > 10:
        hgt = hgt / 1000
        hgt_range = hgt_range / 1000

    for k in range(n_dates):
        ix_points_k = ix_points.copy()
        nan_mask = np.isnan(phuw[:, k])
        valid_points = ~nan_mask
        ix_points_k = np.where(valid_points)[0]
        A = np.column_stack((hgt[ix_points_k], np.ones_like(hgt[ix_points_k])))
        coeff = np.linalg.lstsq(A, phuw[ix_points_k, k], rcond=None)[0]
        ph_tropo_linear[:, k] = np.column_stack((hgt, np.ones_like(hgt))) @ coeff
        ph_tropo_linear[nan_mask, k] = np.nan

    save_h5(workdir, apsname, **{"ph_tropo_linear": ph_tropo_linear})
