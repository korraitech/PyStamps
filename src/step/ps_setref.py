from .utils import read_h5  
from .llh2local import llh2local
import numpy as np
import os

def ps_setref(workdir:str,parms:dict,ps2=None):
    """
    PS_SETREF find reference PS
    """
    psver = int(read_h5(os.path.join(workdir, 'psver.h5'))['psver'])
    psname = f'ps{psver}.h5'

    if ps2 is None:    
        ps2 = read_h5(os.path.join(workdir, psname))
    else:
        ps_temp = read_h5(os.path.join(workdir, psname))
        ps2['ll0'] = ps_temp['ll0']
        ps2['n_ps'] = ps2['lonlat'].shape[0]

    ref_lon = np.array(parms['ref_lon'])
    ref_lat = np.array(parms['ref_lat'])
    ref_centre_lonlat = np.array(parms['ref_centre_lonlat'])
    ref_radius = np.array(parms['ref_radius'])

    if ref_radius == -np.inf:
        ref_ps = 0
    else:
        ref_ps = [i for i, lonlat in enumerate(ps2['lonlat']) if ref_lon[0] < lonlat[0] < ref_lon[0] and ref_lat[1] < lonlat[1] < ref_lat[1]]
        if ref_radius < np.inf:
            ref_xy = llh2local(ref_centre_lonlat, ps2['ll0']) * 1000
            xy = llh2local(ps2['lonlat'][ref_ps, :], ps2['ll0']) * 1000
            dist_sq = (xy[0, :] - ref_xy[0])**2 + (xy[1, :] - ref_xy[1])**2
            ref_ps = [ref_ps[i] for i in range(len(ref_ps)) if dist_sq[i] <= ref_radius**2]

    if not ref_ps:
        if ps2 is not None:
            print('None of your external data points have a reference, all are set as reference.')
            ref_ps = list(range(int(ps2['n_ps'])))

    if ps2 is None:
        if ref_ps == 0:
            print('No reference set')
        else:
            print(f'{len(ref_ps)} ref PS selected')

    return ref_ps
