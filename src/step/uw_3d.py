import numpy as np
from .uw_unwrap_from_grid import uw_unwrap_from_grid
from .uw_grid_wrapped import uw_grid_wrapped
from .uw_stat_costs import uw_stat_costs
from .uw_interp import uw_interp
from .uw_sb_unwrap_space_time import uw_sb_unwrap_space_time

def uw_3d(ph, xy, day, ifgday_ix, bperp=None, options=None):
    """
    Translate of the MATLAB function uw_3d into Python.
    --------------------------------------------------
    Unwrap phase time series (single or multiple master).

    Parameters
    ----------
    ph : numpy.ndarray
        N x M matrix of wrapped phase values (real phase or complex phasor),
        where N is number of pixels and M is number of interferograms.
    xy : numpy.ndarray
        N x 2 matrix of coordinates in meters.
        (Optional extra column can be ignored. If so, first column is ignored.)
    day : numpy.ndarray
        Vector of image acquisition dates in days, relative to master.
    ifgday_ix : numpy.ndarray
        M x 2 matrix giving index to master and slave date in "day"
        for each interferogram.
    bperp : numpy.ndarray or None
        M x 1 vector giving perpendicular baselines (optional).
    options : dict or None
        Dictionary containing optional parameters.

    Returns
    -------
    ph_uw : numpy.ndarray
        Unwrapped phase array.
    msd : any
        Placeholder for whatever output msd represents.
    """


    # ------------------
    # Replicate defaults
    # ------------------
    if bperp is None:
        bperp = np.array([])

    if options is None:
        options = {}

    # Check for malformed options
    valid_options = {
        "la_flag", "scf_flag", "master_day", "grid_size", "prefilt_win",
        "time_win", "unwrap_method", "goldfilt_flag", "lowfilt_flag",
        "gold_alpha", "n_trial_wraps", "temp", "n_temp_wraps", 
        "max_bperp_for_temp_est", "variance", "ph_uw_predef"
    }
    invalid_options = set(options.keys()) - valid_options
    if len(invalid_options) > 0:
        raise ValueError(f'Invalid option(s) detected: {invalid_options}')

    # ----------------
    # Set default vals
    # ----------------
    options.setdefault("master_day", 0)
    options.setdefault("grid_size", 5)
    options.setdefault("prefilt_win", 16)
    options.setdefault("time_win", 365)
    options.setdefault("goldfilt_flag", "n")
    options.setdefault("lowfilt_flag", "n")
    options.setdefault("gold_alpha", 0.8)
    options.setdefault("n_trial_wraps", 6)
    options.setdefault("la_flag", "y")
    options.setdefault("scf_flag", "y")
    options.setdefault("temp", np.array([]))
    options.setdefault("n_temp_wraps", 2)
    options.setdefault("max_bperp_for_temp_est", 100)
    options.setdefault("variance", np.array([]))
    options.setdefault("ph_uw_predef", np.array([]))

    # Convert options.temp to numpy if not empty
    if (options["temp"].size != 0) and (len(options["temp"]) != ph.shape[1]):
        raise ValueError("options['temp'] must be an array of length M (number of ifgs)")

    # -----------
    # Input checks
    # -----------
    if day.ndim == 1:
        day = day.reshape(-1)  # ensure column-like if needed

    # If no ifgday_ix is provided or if it's single-master
    if ifgday_ix.size == 0:
        single_master_flag = 1
    else:
        single_master_flag = 0

    # If unwrap_method not given, pick default
    if "unwrap_method" not in options:
        if single_master_flag == 1:
            options["unwrap_method"] = "3D"
        else:
            options["unwrap_method"] = "3D_FULL"

    # If xy is N x 2, replicate the logic that moves columns
    if xy.shape[1] == 2:
        # replicate MATLAB approach: for an N x 2,
        # we treat columns [2,3] = old [1,2]
        # effectively just shifting to maintain shape
        xy2 = np.zeros((xy.shape[0], 3))
        xy2[:, 1:3] = xy[:, 0:2]
        xy = xy2

    # If we see '3D' or '3D_NEW' but repeated master is missing,
    # force '3D_FULL' or set lowfilt_flag
    if options["unwrap_method"].upper() in ["3D", "3D_NEW"]:
        if len(np.unique(ifgday_ix[:, 0])) == 1:
            options["unwrap_method"] = "3D_FULL"
        else:
            options["lowfilt_flag"] = "y"

    # --------------------------------------------------
    # Match the MATLAB calls in the main function
    # --------------------------------------------------
    uw_grid_wrapped(
        ph,
        xy,
        options["grid_size"],
        options["prefilt_win"],
        options["goldfilt_flag"],
        options["lowfilt_flag"],
        options["gold_alpha"],
        options["ph_uw_predef"],
    )

    # We cleared ph in MATLAB after call, so you might do:
    # del ph
    # but in Python, we can keep or discard. We'll just keep it.

    uw_interp()

    uw_sb_unwrap_space_time(
        day,
        ifgday_ix,
        options["unwrap_method"],
        options["time_win"],
        options["la_flag"],
        bperp,
        options["n_trial_wraps"],
        options["prefilt_win"],
        options["scf_flag"],
        options["temp"],
        options["n_temp_wraps"],
        options["max_bperp_for_temp_est"],
    )

    uw_stat_costs(options["unwrap_method"], options["variance"])

    ph_uw, msd = uw_unwrap_from_grid(xy, options["grid_size"])

    # For timing, MATLAB uses tic/toc. In Python, use time.time() if needed.

    return ph_uw, msd
