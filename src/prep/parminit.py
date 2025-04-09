import os
import json
import numpy
from ..logger import appLogger
from ..misc import get_module_info

def ps_parms_init(workdir:str, rslc_par:dict, parmfile:str = 'parms.json'):
    appLogger.info(">>>>>>>>>>>>>>>> {}".format(get_module_info()))

    # Initialize parameters
    parms = {}
    
    # Set default parameters if not already set
    parms.setdefault('insar_processor', 'snap')
    parms.setdefault('max_topo_err', 20)
    parms.setdefault('quick_est_gamma_flag', 'y')
    parms.setdefault('select_reest_gamma_flag', 'y')
    parms.setdefault('filter_grid_size', 50)
    parms.setdefault('filter_weighting', 'P-square')
    parms.setdefault('gamma_change_convergence', 0.005)
    parms.setdefault('gamma_max_iterations', 3)
    parms.setdefault('slc_osf', 1)
    parms.setdefault('clap_win', 32)
    parms.setdefault('clap_low_pass_wavelength', 800)
    parms.setdefault('clap_alpha', 1)
    parms.setdefault('clap_beta', 0.3)
    parms.setdefault('select_method', 'DENSITY')
    parms.setdefault('density_rand', 20)
    parms.setdefault('percent_rand', 20)
    parms.setdefault('gamma_stdev_reject', 0)
    parms.setdefault('weed_time_win', 730)
    parms.setdefault('weed_max_noise', numpy.inf)
    parms.setdefault('weed_standard_dev', 1.0)
    parms.setdefault('weed_zero_elevation', 'n')
    parms.setdefault('weed_neighbours', 'n')
    parms.setdefault('unwrap_method', '3D')
    parms.setdefault('unwrap_patch_phase', 'n')
    parms.setdefault('drop_ifg_index', [])
    parms.setdefault('unwrap_la_error_flag', 'y')
    parms.setdefault('unwrap_spatial_cost_func_flag', 'n')
    parms.setdefault('unwrap_prefilter_flag', 'y')
    parms.setdefault('unwrap_grid_size', 200)
    parms.setdefault('unwrap_gold_n_win', 32)
    parms.setdefault('unwrap_alpha', 8)
    parms.setdefault('unwrap_time_win', 730)
    parms.setdefault('unwrap_gold_alpha', 0.8)
    parms.setdefault('unwrap_hold_good_values', 'n')
    parms.setdefault('scla_drop_index', [])
    parms.setdefault('scn_wavelength', 100)
    parms.setdefault('scn_time_win', 365)
    parms.setdefault('scn_deramp_ifg', [])
    parms.setdefault('scn_kriging_flag', 'n')
    parms.setdefault('ref_lon', [-numpy.inf, numpy.inf])
    parms.setdefault('ref_lat', [-numpy.inf, numpy.inf])
    parms.setdefault('ref_centre_lonlat', [0, 0])
    parms.setdefault('ref_radius', numpy.inf)
    parms.setdefault('ref_velocity', 0)
    parms.setdefault('n_cores', 1)
    parms.setdefault('plot_dem_posting', 90)
    parms.setdefault('plot_scatterer_size', 120)
    parms.setdefault('plot_pixels_scatterer', 3)
    parms.setdefault('plot_color_scheme', 'inflation')
    parms.setdefault('shade_rel_angle', [90, 45])
    parms.setdefault('lonlat_offset', [0, 0])
    parms.setdefault('merge_resample_size', 0)
    parms.setdefault('merge_standard_dev', numpy.inf)
    parms.setdefault('scla_method', 'L2')
    parms.setdefault('scla_deramp', 'n')
    parms.setdefault('lambda', numpy.nan)
    parms.setdefault('heading', numpy.nan)
    parms.setdefault('sb_scla_drop_index', [])
    parms.setdefault('insar_processor', 'doris')
    parms.setdefault('subtr_tropo', 'n')
    parms.setdefault('tropo_method', 'a_l')

    # Convert numpy types to native Python types for JSON serialization
    for key, value in parms.items():
        if isinstance(value, numpy.ndarray):
            parms[key] = value.tolist()
        elif isinstance(value, numpy.generic):
            parms[key] = value.item()

    # update parms with kwargs
    parms.update(rslc_par)

    # Save parameters to JSON
    with open(os.path.join(workdir, parmfile), 'w') as f:
        json.dump(parms, f, indent=4)
