from .utils import read_h5  
from .llh2local import llh2local
import numpy as np
import os

def ps_setref(workdir:str,parms:dict):
    """
    PS_SETREF find reference PS
    """
    psver = int(read_h5(os.path.join(workdir, 'psver.h5'))['psver'])
    psname = f'ps{psver}.h5'
    ps2 = read_h5(os.path.join(workdir, psname))

    ref_lon = np.array(parms['ref_lon'])
    ref_lat = np.array(parms['ref_lat'])
    ref_centre_lonlat = np.array(parms['ref_centre_lonlat'])
    ref_radius = np.array(parms['ref_radius'])

    if ref_radius == -np.inf:
        ref_ps = 0
    else:
        ps2_lonlat = ps2['lonlat']
        ref_ps = np.where(
            (ps2_lonlat[:, 0] > ref_lon[0]) & 
            (ps2_lonlat[:, 0] < ref_lon[1]) & 
            (ps2_lonlat[:, 1] > ref_lat[0]) & 
            (ps2_lonlat[:, 1] < ref_lat[1])
        )[0]
        if ref_radius < np.inf:
            ref_xy = llh2local(ref_centre_lonlat, ps2['ll0']) * 1000
            xy = llh2local(ps2['lonlat'][ref_ps, :], ps2['ll0']) * 1000
            dist_sq = (xy[0, :] - ref_xy[0])**2 + (xy[1, :] - ref_xy[1])**2
            ref_ps = [ref_ps[i] for i in range(len(ref_ps)) if dist_sq[i] <= ref_radius**2]

    return ref_ps
