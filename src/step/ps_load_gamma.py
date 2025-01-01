import numpy as np
import os
import h5py
from datetime import datetime
from .ps_llh_local import ps_llh_local
from .utils import read_lines,get_par
from .stamps_save import stamps_save

def ps_load_gamma(workdir:str,patch:str, endian='b'):
    """
    Initial load of files into Python workspace.
    
    Parameters:
        workdir (str): Path to directory containing input / output files
        endian (str): Endianness of binary files ('b' for big-endian, 'l' for little-endian)
    """
    print("Running Step-01 ...\t[{}]".format(patch))
    patch_dir = os.path.join(workdir,patch)

    # Set endian format for numpy
    endian_fmt = '>' if endian == 'b' else '<'
    
    # Files inside patch
    phname = os.path.join(patch_dir, 'pscands.1.ph')
    ijname = os.path.join(patch_dir, 'pscands.1.ij')
    llname = os.path.join(patch_dir, 'pscands.1.ll')
    hgtname = os.path.join(patch_dir, 'pscands.1.hgt')
    daname = os.path.join(patch_dir, 'pscands.1.da')

    # Files in parent directory
    rscname = os.path.join(workdir, 'rsc.txt')
    pscname = os.path.join(workdir, 'pscphase.in')
    
    psver = 1
    
    rslcpar = read_lines(rscname)[0]
    ifgs = read_lines(pscname)[1:]
    
    # Process dates
    nb = len(ifgs[0])
    master_day = int(ifgs[0][nb-22:nb-14])
    n_ifg = len(ifgs)
    n_image = n_ifg
    
    # Convert dates to datetime objects
    days = []
    for ifg in ifgs:
        date_str = ifg[nb-13:nb-5]
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        days.append(datetime(year, month, day))
    
    #master_day_yyyymmdd = master_day
    year = master_day // 10000
    month = (master_day - year * 10000) // 100
    monthday = master_day - year * 10000 - month * 100
    master_date = datetime(year, month, monthday)
    
    # Find master index
    master_ix = sum(1 for d in days if d < master_date)
    if days[master_ix] != master_date:
        master_master_flag = '0'  # no null master-master ifg provided
        days.insert(master_ix, master_date)
    else:
        master_master_flag = '1'  # yes, null master-master ifg provided
    
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
    ij = np.loadtxt(ijname)
    n_ps = ij.shape[0]
    
    # Calculate geometry
    mean_az = naz/2 - 0.5  # mean azimuth line
    rg = rgn + ij[:, 2] * rps
    look = np.arccos((se**2 + rg**2 - re**2)/(2*se*rg))  # Satellite look angles
    
    # Initialize baseline matrix
    bperp_mat = np.zeros((n_ps, n_image), dtype=np.float32)
    
    # Process baselines
    for i in range(n_ifg):
        param_fields = get_par(f"{ifgs[i][:nb-4]}base")
        B_TCN  = [float(par) for par in param_fields["initial_baseline(TCN)"][:3]]
        BR_TCN  = [float(par) for par in param_fields["initial_baseline_rate"][:3]]
        
        bc = B_TCN[1] + BR_TCN[1]*(ij[:, 1] - mean_az)/prf
        bn = B_TCN[2] + BR_TCN[2]*(ij[:, 1] - mean_az)/prf
        bperp_mat[:, i] = bc*np.cos(look) - bn*np.sin(look)
    
    # Calculate mean perpendicular baseline
    bperp = np.mean(bperp_mat, axis=0)
    
    # Adjust baseline matrix based on master
    if master_master_flag == '1':
        bperp_mat = np.delete(bperp_mat, master_ix, axis=1)
    else:
        bperp = np.insert(bperp, master_ix, 0)
    
    # Calculate incidence angles
    inci = np.arccos((se**2 - re**2 - rg**2)/(2*re*rg))
    mean_incidence = np.mean(inci)
    mean_range = rgc
    
    # Read phase data
    ph = np.zeros((n_ps, n_ifg), dtype=np.complex64)
    with open(phname, 'rb') as fid:
        for i in range(n_ifg):
            # Read all data for one interferogram at once
            ph_bit = np.fromfile(fid, dtype=f'{endian_fmt}f', count=n_ps*2)
            ph[:, i] = ph_bit[::2] + 1j*ph_bit[1::2]
    
    # Process zero phases
    zero_ph = np.sum(ph == 0, axis=1)
    #nonzero_ix = zero_ph <= 1
    
    if master_master_flag == '1':
        ph[:, master_ix] = 1
    else:
        ph = np.insert(ph, master_ix, np.ones(n_ps), axis=1)
        n_ifg += 1
        n_image += 1
    
    # Read lon/lat data
    if os.path.exists(llname):
        lonlat = np.fromfile(llname, dtype=f'{endian_fmt}f').reshape(-1, 2).astype(np.float64)
    else:
        raise FileNotFoundError(f"{llname} does not exist")
    
    # Calculate center point
    ll0 = (np.max(lonlat, axis=0) + np.min(lonlat, axis=0)) / 2
    
    # Convert to local coordinates
    xy = ps_llh_local(lonlat.T, ll0).T * 1000
    
    # # Sort coordinates and find corners
    # sort_x = xy[xy[:, 0].argsort()]
    # sort_y = xy[xy[:, 1].argsort()]
    # n_pc = round(n_ps * 0.001)
    # bl = np.mean(sort_x[:n_pc], axis=0)  # bottom left corner
    # tr = np.mean(sort_x[-n_pc:], axis=0)  # top right corner
    # br = np.mean(sort_y[:n_pc], axis=0)  # bottom right corner
    # tl = np.mean(sort_y[-n_pc:], axis=0)  # top left corner
    
    # Rotate coordinates
    theta = (180 - heading) * np.pi / 180
    if theta > np.pi:
        theta -= 2 * np.pi
    
    rotm = np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])
    
    xy = xy.T  # Transpose to match MATLAB's operation
    xynew = rotm @ xy  # rotate so that scene axes approx align with x=0 and y=0
    
    # Check rotation improvement using MATLAB's logic
    if (max(xynew[0]) - min(xynew[0]) < max(xy[0]) - min(xy[0]) and
        max(xynew[1]) - min(xynew[1]) < max(xy[1]) - min(xy[1])):
        xy = xynew  # check that rotation is an improvement
        print(f'Rotating by {theta * 180/np.pi} degrees')
    
    xy = xy.T  # Transpose back
    
    # Convert to single precision and round to millimeters
    xy = xy.astype(np.float32)
    sort_ix = np.lexsort((xy[:, 0], xy[:, 1]))  # sort in ascending y order
    xy = xy[sort_ix]
    xy = np.round(xy * 1000) / 1000  # round to mm
    
    # Add index column
    xy = np.column_stack((np.arange(1, n_ps + 1), xy))
    
    # Sort other arrays
    ph = ph[sort_ix]
    ij = ij[sort_ix]
    ij[:, 0] = np.arange(1, n_ps + 1)
    lonlat = lonlat[sort_ix]
    bperp_mat = bperp_mat[sort_ix]
    
    # Remove NaN values if present in lonlat, phase
    ix_nan = np.any(np.isnan(lonlat), axis=1) | np.any(np.isnan(ph), axis=1)
    lonlat = lonlat[~ix_nan]
    ij = ij[~ix_nan]
    xy = xy[~ix_nan]
    n_ps = len(lonlat)
    
    # Update indices
    ij[:, 0] = np.arange(1, n_ps + 1)
    xy[:, 0] = np.arange(1, n_ps + 1)

    with h5py.File(os.path.join(patch_dir, 'psver.h5'), 'w') as f:
        f.create_dataset('psver', data=psver)
    
    # Save results with output directory
    savename = f'ps{psver}.h5'
    days_str = np.array([d.strftime('%Y-%m-%d %H:%M:%S') for d in days])
    days_str = days_str.astype('S')
    master_date_str = np.array([master_date.strftime('%Y-%m-%d %H:%M:%S')])
    master_date_str = master_date_str.astype('S')
    stamps_save(patch_dir,savename,
        **{"ij":ij, "lonlat":lonlat, "xy":xy, "bperp":bperp, "days_str":days_str, 
         "master_date_str":master_date_str, "master_ix":master_ix,"n_ifg":n_ifg, 
         "n_image":n_image, "n_ps":n_ps, "sort_ix":sort_ix, "ll0":ll0, 
         "master_ix":master_ix, "mean_incidence":mean_incidence, "mean_range":mean_range})
    
    # Save phase data
    phsavename = f'ph{psver}.h5'
    ph = ph[~ix_nan]
    stamps_save(patch_dir,phsavename, **{"ph":ph})
    
    # Save baseline data
    bpsavename = f'bp{psver}.h5'
    bperp_mat = bperp_mat[~ix_nan]
    stamps_save(patch_dir, bpsavename, **{"bperp_mat":bperp_mat})
    
    # Save look angle data
    lasavename = f'la{psver}.h5'
    la = inci[sort_ix]  # store incidence not look angle for gamma
    la = la[~ix_nan]
    stamps_save(patch_dir, lasavename, **{"la":la})
    
    # Save D_A if exists
    if os.path.exists(daname):
        D_A = np.loadtxt(daname)
        D_A = D_A[sort_ix]
        D_A = D_A[~ix_nan]
        dasavename = f'da{psver}.h5'
        stamps_save(patch_dir, dasavename, **{"D_A":D_A})
    
    # Save height if exists
    if os.path.exists(hgtname):
        with open(hgtname, 'rb') as fid:
            hgt = np.fromfile(fid, dtype=f'{endian_fmt}f')
        hgt = hgt[sort_ix]
        hgt = hgt[~ix_nan]
        hgtsavename = f'hgt{psver}.h5'
        stamps_save(patch_dir, hgtsavename, **{"hgt":hgt})

