import numpy as np
import os
from .utils import read_h5, save_h5

def step_aps_linear(workdir: str):
    """
    Compute tropospheric delay map from linear relation between phase and topography.
    
    Args:
        workdir (str): Path where output files will be saved. Defaults to current directory.
        patch (str): Patch name.
        parms (dict): Parameters dictionary.
    """
    print("Running Step-aps_linear ...")
    print('Computing linear relation between phase and topography...')

    psver = int(read_h5(os.path.join(workdir,'psver.h5'))['psver'])

    phuw = read_h5(os.path.join(workdir, f'phuw{psver}.h5'))['ph_uw']
    n_dates = phuw.shape[1]

    hgt = read_h5(os.path.join(workdir, f'hgt{psver}.h5'))['hgt']
    ix_points = np.arange(hgt.shape[0])

    # Compute linear relation between phase and topography
    ph_tropo_linear = np.zeros((hgt.shape[0], n_dates))
    hgt_range = np.array([np.min(hgt), np.max(hgt)])
    if hgt_range[1] > 10:
        hgt = hgt / 1000
        hgt_range = hgt_range / 1000

    # Main computation loop
    for k in range(n_dates):
        ix_points_k = ix_points.copy()
        
        # Remove NaN phases
        nan_mask = np.isnan(phuw[:, k])
        valid_points = ~nan_mask
        ix_points_k = np.where(valid_points)[0]
        
        # Design matrix
        A = np.column_stack((hgt[ix_points_k], np.ones_like(hgt[ix_points_k])))
        
        # Compute linear relation
        coeff = np.linalg.lstsq(A, phuw[ix_points_k, k], rcond=None)[0]
        
        # Compute delay
        ph_tropo_linear[:, k] = np.column_stack((hgt, np.ones_like(hgt))) @ coeff
        
        # Restore NaN values
        ph_tropo_linear[nan_mask, k] = np.nan

    # Save result
    apsname = os.path.join(workdir, f'tca{psver}.h5')
    save_h5(workdir, apsname, **{"ph_tropo_linear": ph_tropo_linear})
