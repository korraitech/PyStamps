import numpy
import h5py
import os
from datetime import datetime
from .llh2local import llh2local
from .utils import read_lines, get_par, read_h5, save_h5
from ..misc import get_module_info
from ..logger import appLogger
from numba import jit, prange, set_num_threads


# @jit(nopython=True, parallel=True)
def _process_dates(date_strings, nb):
    """
    Helper function to process dates (without numba, to avoid 
    slicing issues on Unicode arrays).
    Extracts year, month, day from each date string.
    """
    n = len(date_strings)
    years = numpy.zeros(n, dtype=numpy.int32)
    months = numpy.zeros(n, dtype=numpy.int32)
    days = numpy.zeros(n, dtype=numpy.int32)
    
    for i in range(n):
        date_str = date_strings[i][nb-13:nb-5]
        years[i] = int(date_str[:4])
        months[i] = int(date_str[4:6])
        days[i] = int(date_str[6:8])
    
    return years, months, days

@jit(nopython=True, parallel=True)
def compute_rg_look(ij, rgn, rps, se, re):
    """
    Compute slant range (rg) and look angle for each PS point in parallel.
    """
    n_ps = ij.shape[0]
    rg = numpy.empty(n_ps, dtype=numpy.float64)
    look = numpy.empty(n_ps, dtype=numpy.float64)
    for i in prange(n_ps):
        rg[i] = rgn + ij[i, 2] * rps
        # Satellite look angles
        look[i] = numpy.arccos((se**2 + rg[i]**2 - re**2) / (2.0 * se * rg[i]))
    return rg, look

@jit(nopython=True, parallel=True)
def compute_inci(se, re, rg):
    """
    Compute incidence angle for each PS point in parallel.
    """
    n_ps = rg.shape[0]
    inci = numpy.empty(n_ps, dtype=numpy.float64)
    for i in prange(n_ps):
        # incidence (gamma) angle calculation
        inci[i] = numpy.arccos((se**2 - re**2 - rg[i]**2) / (2.0 * re * rg[i]))
    return inci

@jit(nopython=True, parallel=True)
def compute_bperp_mat(ij, mean_az, prf, look, B_TCN_array, BR_TCN_array):
    """
    Compute perpendicular baseline for each PS point in parallel,
    given B_TCN and BR_TCN arrays.
    """
    n_ps = ij.shape[0]
    n_ifg = B_TCN_array.shape[0]
    bperp_mat = numpy.empty((n_ps, n_ifg), dtype=numpy.float32)
    
    for i in prange(n_ifg):
        bc = B_TCN_array[i, 1] + BR_TCN_array[i, 1]*(ij[:, 1] - mean_az)/prf
        bn = B_TCN_array[i, 2] + BR_TCN_array[i, 2]*(ij[:, 1] - mean_az)/prf
        bperp_mat[:, i] = bc*numpy.cos(look) - bn*numpy.sin(look)
        
    return bperp_mat

@jit(nopython=True, parallel=True)
def _fill_baseline_arrays(n_ifg, temp_B_array, temp_BR_array, B_TCN_array, BR_TCN_array):
    """
    Copy previously read float parameters into the final TCN arrays in parallel.
    This part can be accelerated by Numba because it is just numeric copying.
    """
    for i in prange(n_ifg):
        for j in range(3):
            B_TCN_array[i, j] = temp_B_array[i, j]
            BR_TCN_array[i, j] = temp_BR_array[i, j]
    return B_TCN_array, BR_TCN_array


def _read_all_baselines(ifgs, nb):
    """
    Read the baseline parameters from each .base file in a normal Python loop.
    The main cost here is file I/O, so Numba won't help—this is purely Python I/O.
    """
    n_ifg = len(ifgs)
    # Temporary arrays for storing float data before passing to the numba function.
    temp_B_array = numpy.empty((n_ifg, 3), dtype=numpy.float32)
    temp_BR_array = numpy.empty((n_ifg, 3), dtype=numpy.float32)

    for i in range(n_ifg):
        b_path = f"{ifgs[i][:nb-4]}base"  # e.g., "20200102base"
        param_fields_ifg = get_par(b_path)
        # Convert string parameters to float
        baseline_TCN = [float(x) for x in param_fields_ifg["initial_baseline(TCN)"][:3]]
        baseline_rate = [float(x) for x in param_fields_ifg["initial_baseline_rate"][:3]]
        temp_B_array[i] = baseline_TCN
        temp_BR_array[i] = baseline_rate

    return temp_B_array, temp_BR_array


def step_1_ps_load_gamma(workdir: str, patch: str, num_threads: int = 1):
    """
    Initial load of files into Python workspace, with added parallelization via Numba.

    Parameters:
        workdir (str): Path to directory containing input / output files
        patch (str): Patch identifier
        num_threads (int): Number of threads to instruct Numba to use (default: 1).
    """
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {} {} || {}".format(
            get_module_info(),workdir, patch, num_threads)
    )
    # Configure Numba parallel threads:
    set_num_threads(num_threads)

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
    date_strings = numpy.array([ifg for ifg in ifgs], dtype='<U100')  # ensure uniform dtype
    years, months, days_arr = _process_dates(date_strings, nb)
    days = [datetime(years[i], months[i], days_arr[i]) for i in range(len(years))]
    
    year = master_day // 10000
    month = (master_day - year * 10000) // 100
    monthday = master_day - year * 10000 - month * 100
    master_date = datetime(year, month, monthday)
    
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
    
    # Calculate geometry in parallel
    mean_az = naz / 2 - 0.5
    rg, look = compute_rg_look(ij, rgn, rps, se, re)
    
    # 1) Read all baseline parameters at once (I/O bound)
    temp_B_array, temp_BR_array = _read_all_baselines(ifgs, nb)

    # 2) Allocate final arrays
    n_ifg = len(ifgs)
    B_TCN_array = numpy.zeros((n_ifg, 3), dtype=numpy.float32)
    BR_TCN_array = numpy.zeros((n_ifg, 3), dtype=numpy.float32)

    # 3) Use a Numba-accelerated function to fill those final arrays
    _fill_baseline_arrays(n_ifg, temp_B_array, temp_BR_array, B_TCN_array, BR_TCN_array)
    
    # Compute baselines in parallel
    bperp_mat = compute_bperp_mat(ij, mean_az, prf, look, B_TCN_array, BR_TCN_array)
    
    # Calculate mean perpendicular baseline
    bperp = numpy.mean(bperp_mat, axis=0)
    
    # Adjust baseline matrix based on master
    if master_master_flag == '1':
        bperp_mat = numpy.delete(bperp_mat, master_ix, axis=1)
    else:
        bperp = numpy.insert(bperp, master_ix, 0)
    
    # Calculate incidence angles in parallel
    inci = compute_inci(se, re, rg)
    mean_incidence = numpy.mean(inci)
    mean_range = rgc
    
    # Read phase data
    ph = read_h5(phpath)["data"]
    if master_master_flag == '1':
        ph[:, master_ix] = 1
    else:
        ph = numpy.insert(ph, master_ix, numpy.ones(n_ps), axis=1)
        n_ifg += 1
        n_image += 1
    
    # Read lon/lat data
    lonlat = read_h5(llpath)["data"]
    
    # Calculate center point
    ll0 = (numpy.max(lonlat, axis=0) + numpy.min(lonlat, axis=0)) / 2
    
    # Convert to local coordinates
    xy = llh2local(lonlat.T, ll0).T * 1000
    
    # Rotate coordinates
    theta = (180 - heading) * numpy.pi / 180
    if theta > numpy.pi:
        theta -= 2 * numpy.pi
    
    rotm = numpy.array([
        [numpy.cos(theta), numpy.sin(theta)],
        [-numpy.sin(theta), numpy.cos(theta)]
    ])
    
    xy_trans = xy.T
    xynew = rotm @ xy_trans
    
    # Check rotation improvement
    if ((numpy.max(xynew[0]) - numpy.min(xynew[0]) <
         numpy.max(xy_trans[0]) - numpy.min(xy_trans[0])) and
        (numpy.max(xynew[1]) - numpy.min(xynew[1]) <
         numpy.max(xy_trans[1]) - numpy.min(xy_trans[1]))):
        xy = xynew.T
        print(f'Rotating by {theta * 180/numpy.pi} degrees')
    else:
        xy = xy_trans.T
    
    # Convert to single precision and round
    xy = xy.astype(numpy.float32)
    sort_ix = numpy.lexsort((xy[:, 0], xy[:, 1]))
    xy = xy[sort_ix]
    xy = numpy.round(xy * 1000) / 1000
    
    # Add index column
    xy = numpy.column_stack((numpy.arange(1, n_ps + 1), xy))
    
    # Sort other arrays accordingly
    ph = ph[sort_ix]
    ij = ij[sort_ix]
    ij[:, 0] = numpy.arange(1, n_ps + 1)
    lonlat = lonlat[sort_ix]
    bperp_mat = bperp_mat[sort_ix]
    
    # Remove NaN values if present
    ix_nan = numpy.any(numpy.isnan(lonlat), axis=1) | numpy.any(numpy.isnan(ph), axis=1)
    lonlat = lonlat[~ix_nan]
    ij = ij[~ix_nan]
    xy = xy[~ix_nan]
    n_ps = len(lonlat)
    
    # Update indices
    ij[:, 0] = numpy.arange(1, n_ps + 1)
    xy[:, 0] = numpy.arange(1, n_ps + 1)

    psver = 1
    with h5py.File(os.path.join(patch_dir, 'psver.h5'), 'w') as f:
        f.create_dataset('psver', data=psver)
    
    # Save results
    savename = f'ps{psver}.h5'
    days_str = numpy.array([d.strftime('%Y-%m-%d %H:%M:%S') for d in days])
    days_str = days_str.astype('S')
    master_date_str = numpy.array([master_date.strftime('%Y-%m-%d %H:%M:%S')])
    master_date_str = master_date_str.astype('S')
    
    save_h5(
        patch_dir, savename,
        **{
            "ij": ij,
            "lonlat": lonlat,
            "xy": xy,
            "bperp": bperp,
            "days_str": days_str,
            "master_date_str": master_date_str,
            "master_ix": master_ix,
            "n_ifg": n_ifg,
            "n_image": n_image,
            "n_ps": n_ps,
            "sort_ix": sort_ix,
            "ll0": ll0,
            "master_ix": master_ix,
            "mean_incidence": mean_incidence,
            "mean_range": mean_range
        }
    )
    
    # Save phase data
    phsavename = f'ph{psver}.h5'
    ph = ph[~ix_nan]
    save_h5(patch_dir, phsavename, **{"ph": ph})
    
    # Save baseline data
    bpsavename = f'bp{psver}.h5'
    bperp_mat = bperp_mat[~ix_nan]
    save_h5(patch_dir, bpsavename, **{"bperp_mat": bperp_mat})
    
    # Save look angle data
    lasavename = f'la{psver}.h5'
    la = inci[sort_ix]
    la = la[~ix_nan]
    save_h5(patch_dir, lasavename, **{"la": la})
    
    # Save D_A if exists
    D_A = read_h5(dapath)["data"]
    D_A = D_A[sort_ix]
    D_A = D_A[~ix_nan]
    dasavename = f'da{psver}.h5'
    save_h5(patch_dir, dasavename, **{"D_A": D_A})
    
    # Save height if exists
    hgt = read_h5(htpath)["data"]
    hgt = hgt[sort_ix]
    hgt = hgt[~ix_nan]
    hgtsavename = f'hgt{psver}.h5'
    save_h5(patch_dir, hgtsavename, **{"hgt": hgt})
