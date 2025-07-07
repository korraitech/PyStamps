import numpy as np
import os
from datetime import datetime,timezone
from .llh2local import llh2local
from .utils import read_lines, get_par, read_h5, save_h5
from ..misc import get_module_info
from ..logger import appLogger

def step_1_ps_loadgm(workdir: str, patch: str, num_threads: int = 1):
    """
    Initial load of files into HDF5 format in python workspace

    Parameters:
        workdir (str): Path to directory containing input / output files
        patch (str): Patch identifier
        num_threads (int): Number of threads to instruct Numba to use (default: 1).
    """
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {} {} || {}".format(
            get_module_info(),workdir, patch, num_threads)
    )

    patch_dir = os.path.join(workdir, patch)
    
    # Files inside patch
    phpath = os.path.join(patch_dir, 'pscands_ph.h5')
    ijpath = os.path.join(patch_dir, 'pscands_ij.h5')
    llpath = os.path.join(patch_dir, 'pscands_ll.h5')
    htpath = os.path.join(patch_dir, 'pscands_ht.h5')
    dapath = os.path.join(patch_dir, 'pscands_da.h5')

    # Files in parent directory
    rscpath = os.path.join(workdir, 'rsc.txt')
    pscpath = os.path.join(workdir, 'pscphase.in')
    
    rslcpar = read_lines(rscpath)[0]
    ifgs = read_lines(pscpath)[1:]
    
    # Process dates
    nb = len(ifgs[0])
    master_day = int(ifgs[0][nb-22:nb-14])
    n_ifg = len(ifgs)
    n_image = n_ifg
    
    # Convert dates to datetime objects
    date_strings = np.array([ifg for ifg in ifgs], dtype='<U100')
    years = [int(s[nb - 13 : nb - 9]) for s in date_strings]
    months = [int(s[nb - 9 : nb - 7]) for s in date_strings]
    days = [int(s[nb - 7: nb - 5]) for s in date_strings]

    days = [datetime(years[i], months[i], days[i], tzinfo=timezone.utc) for i in range(len(years))]
    
    year = master_day // 10000
    month = (master_day - year * 10000) // 100
    monthday = master_day - year * 10000 - month * 100
    master_date = datetime(year, month, monthday, tzinfo=timezone.utc)
    
    # Find master index
    master_ix = sum(1 for d in days if d < master_date)
    if days[master_ix] != master_date:
        master_master_flag = '0'
        days.insert(master_ix, master_date)
    else:
        master_master_flag = '1'
    
    # Read heading
    param_fields = get_par(rslcpar)
    heading  = float(param_fields["heading"][0])

    # Read radar parameters
    rps = int(param_fields["range_pixel_spacing"][0])
    rgn = float(param_fields["near_range_slc"][0])
    se = float(param_fields["sar_to_earth_center"][0])
    re = float(param_fields["earth_radius_below_sensor"][0])
    rgc = float(param_fields["center_range_slc"][0])
    naz = int(param_fields["azimuth_lines"][0])
    prf = float(param_fields["prf"][0])
    
    # Load IJ coordinates
    ij = read_h5(ijpath)["data"]
    n_ps = ij.shape[0]
    
    mean_az = naz / 2 - 0.5
    rg = rgn + ij[:, 2] * rps
    look = np.arccos((se**2 + rg**2 - re**2) / (2 * se * rg))
    
    bperp_mat = np.empty((n_ps, n_ifg), dtype=np.float64)
    for i in range(n_ifg):
        base_path = f"{ifgs[i][:nb-4]}base"
        base_param = get_par(base_path)
        bc=float(base_param["initial_baseline(TCN)"][1]) + float(base_param["initial_baseline_rate"][1])*(ij[:,1]-mean_az)/prf
        bn=float(base_param["initial_baseline(TCN)"][2]) + float(base_param["initial_baseline_rate"][2])*(ij[:,1]-mean_az)/prf
        bperp_mat[:,i]=bc*np.cos(look)-bn*np.sin(look)

    bperp = np.nanmean(bperp_mat, axis=0)
    if master_master_flag == '1':
        bperp_mat = np.delete(bperp_mat, master_ix, axis=1)
    else:
        bperp = np.insert(bperp, master_ix, 0)
    
    # Calculate incidence angles in parallel
    inci = np.arccos((se**2 - re**2 - rg**2) / (2 * re * rg))
    mean_incidence = np.nanmean(inci)
    mean_range = rgc
    
    # Read phase data
    ph = read_h5(phpath)["data"]
    if master_master_flag == '1':
        ph[:, master_ix] = 1
    else:
        ph = np.insert(ph, master_ix, np.ones(n_ps), axis=1)
        n_ifg += 1
        n_image += 1
    
    # Read lon/lat data
    lonlat = read_h5(llpath)["data"]

    # Calculate center point
    ll0 = (np.nanmax(lonlat, axis=0) + np.nanmin(lonlat, axis=0)) / 2
    
    # Convert to local coordinates
    xy = llh2local(lonlat.T, ll0).T * 1000
    
    # Rotate coordinates
    theta = (180 - heading) * np.pi / 180
    if theta > np.pi:
        theta -= 2 * np.pi
    
    rotm = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    
    xy_trans = xy.T
    xynew = rotm @ xy_trans
    
    # Check rotation improvement
    if ((np.nanmax(xynew[0]) - np.nanmin(xynew[0]) <
         np.nanmax(xy_trans[0]) - np.nanmin(xy_trans[0])) and
        (np.nanmax(xynew[1]) - np.nanmin(xynew[1]) <
         np.nanmax(xy_trans[1]) - np.nanmin(xy_trans[1]))):
        xy = xynew.T
        print(f'Rotating by {theta * 180/np.pi} degrees')
    else:
        xy = xy_trans.T
    
    # Convert to single precision and round
    sort_ix = np.lexsort((xy[:, 0], xy[:, 1]))
    xy = xy[sort_ix]
    xy = np.round(xy * 1000) / 1000

    # Sort other arrays accordingly
    ph = ph[sort_ix]
    ij = ij[sort_ix]
    inci = inci[sort_ix]
    lonlat = lonlat[sort_ix]
    bperp_mat = bperp_mat[sort_ix]

    # Find the inds where nan is present
    ix_nan = np.any(np.isnan(lonlat), axis=1) | np.any(np.isnan(ph), axis=1)

    lonlat = lonlat[~ix_nan]
    ij = ij[~ix_nan]
    xy = xy[~ix_nan]
    ph = ph[~ix_nan]
    la = inci[~ix_nan]
    bperp_mat = bperp_mat[~ix_nan]
    n_ps = len(lonlat)

    # Update indices
    ij[:, 0] = np.arange(n_ps)
    xy = np.column_stack((np.arange(n_ps), xy))

    psver = 1
    save_h5(patch_dir,'psver.h5',**{'psver':psver})
    
    # Save results
    psname = f'ps{psver}.h5'
    day = np.array([int(d.timestamp()/86400) for d in days])
    master_day = np.array([int(master_date.timestamp()/86400)])
    
    save_h5(
        patch_dir, psname,
        **{
            "ij": ij,
            "lonlat": lonlat,
            "xy": xy,
            "bperp": bperp,
            "day": day,
            "master_day": master_day,
            "master_ix": master_ix,
            "n_ifg": n_ifg,
            "n_image": n_image,
            "sort_ix": sort_ix,
            "n_ps": n_ps,
            "ll0": ll0,
            "master_ix": master_ix,
            "mean_incidence": mean_incidence,
            "mean_range": mean_range
        }
    )
    
    # Save phase data
    phname = f'ph{psver}.h5'
    save_h5(patch_dir, phname, **{"ph": ph})
    
    # Save baseline data
    bpname = f'bp{psver}.h5'
    save_h5(patch_dir, bpname, **{"bperp_mat": bperp_mat})
    
    # Save look angle data
    laname = f'la{psver}.h5'
    save_h5(patch_dir, laname, **{"la": la})
    
    # Save D_A if exists
    D_A = read_h5(dapath)["data"]
    D_A = D_A[sort_ix]
    D_A = D_A[~ix_nan]
    daname = f'da{psver}.h5'
    save_h5(patch_dir, daname, **{"D_A": D_A})
    
    # Save height if exists
    hgt = read_h5(htpath)["data"]
    hgt = hgt[sort_ix]
    hgt = hgt[~ix_nan]
    hgtname = f'hgt{psver}.h5'
    save_h5(patch_dir, hgtname, **{"hgt": hgt})
