import numpy as np
import os
from ..misc import get_module_info
from ..logger import appLogger
from .utils import read_h5, save_h5,lscov
from .ps_deramp import ps_deramp
from .ps_setref import ps_setref

def step_8_ps_plot(workdir:str, parms:dict):
    """
    Compute the phase and velocity results.
    """
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {}".format(
            get_module_info(),workdir)
    )

    psver = int(read_h5(os.path.join(workdir, 'psver.h5'))['psver'])
    pmname = f'pm{psver}.h5'
    psname = f'ps{psver}.h5'
    apsname = f'tca{psver}.h5'
    ifgstdname = f'ifgstd{psver}.h5'
    phuwname = f'phuw{psver}.h5'
    sclaname = f'scla{psver}.h5'
    plotname = f'ps_plot.h5'

    pm = read_h5(os.path.join(workdir, pmname))    
    ps = read_h5(os.path.join(workdir, psname))
    uw = read_h5(os.path.join(workdir, phuwname))
    aps = read_h5(os.path.join(workdir, apsname))
    scla = read_h5(os.path.join(workdir, sclaname))
    ifgstd = read_h5(os.path.join(workdir, ifgstdname))

    master_day = ps['master_day']
    n_ps = ps['n_ps']
    n_ifg = ps['n_ifg']
 
    unwrap_ifg_index=np.arange(n_ifg)
    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, ps['master_ix'])

    ph_uw=uw['ph_uw']
    aps_corr = aps['ph_tropo_linear']
    ph_uw = ph_uw - scla['ph_scla'] - aps_corr
    ph_uw, _ = ps_deramp(ps,ph_uw)
    ref_ps=ps_setref(workdir,parms)

    def compute_ph_mm(ph_uw, unwrap_ifg_index, ref_ps, n_ps,day):
        ph_uw = ph_uw - scla['C_ps_uw'].reshape(-1, 1)
        ph_uw = ph_uw[:, unwrap_ifg_index]
        day = day[unwrap_ifg_index]
        ph_uw = ph_uw - np.tile(np.nanmean(ph_uw[ref_ps, :], axis=0), (n_ps, 1))
        return -ph_uw*parms['lambda']*1000/(4*np.pi),day
    
    def compute_ph_disp(ph_uw, unwrap_ifg_index, ref_ps, n_ps):
        ph_uw = ph_uw[:, unwrap_ifg_index]
        ph_uw = ph_uw - np.tile(np.nanmean(ph_uw[ref_ps, :], axis=0), (n_ps, 1))
        ifgvar = (ifgstd['ifg_std'] * np.pi / 181) ** 2
        sm_cov = np.diag(ifgvar[unwrap_ifg_index])
        G = np.column_stack([np.ones(day.shape), day - master_day])
        m=lscov(G,ph_uw.T,np.diag(np.linalg.inv(sm_cov)))
        return -m[1,:]*365.25/4/np.pi*parms['lambda']*1000

    ph_mm,day = compute_ph_mm(ph_uw.copy(), unwrap_ifg_index, ref_ps, n_ps, ps['day'])
    ph_disp = compute_ph_disp(ph_uw.copy(), unwrap_ifg_index, ref_ps, n_ps)

    save_h5(workdir, plotname, **{
        'coh_ps':pm['coh_ps'],'lonlat':ps['lonlat'],
        'day':day,
        'ph_mm':ph_mm,
        'ph_disp':ph_disp,
    })
