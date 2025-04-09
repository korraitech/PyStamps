import os
import json
from .env import POOL_SIZE
from .step.utils import read_lines
from .step.step_1_ps_load_gamma import step_1_ps_load_gamma
from .step.step_2_ps_estm_gamma import step_2_ps_estm_gamma
# from .step.step_3_ps_select import step_3_ps_select
# from .step.step_4_ps_weed import step_4_ps_weed
# from .step.step_5a_ps_correct_phase import step_5a_ps_correct_phase
# from .step.step_5b_ps_merge_patches import step_5b_ps_merge_patches
# from .step.step_aps_linear import step_aps_linear
# from .step.step_6_ps_unwrap import step_6_ps_unwrap
# from .step.step_7a_ps_calc_scla import step_7a_ps_calc_scla
# from .step.step_7b_ps_smooth_scla import step_7b_ps_smooth_scla
from multiprocessing.pool import ThreadPool

def read_json(path:str):
    data = {}
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def patch_task(parmas:dict):
    patch = parmas["patch"]
    parms = parmas["parms"]
    workdir = parmas["workdir"]
    step_1_ps_load_gamma(workdir,patch)
    step_2_ps_estm_gamma(workdir,patch,parms)
    # step_3_ps_select(workdir,patch,parms)
    # step_4_ps_weed(workdir,patch,parms)
    # step_5a_ps_correct_phase(workdir,patch)

def run_stamps_steps(workdir:str):
    patches = read_lines(os.path.join(workdir,"patch.list"))
    parms = read_json(os.path.join(workdir,"parms.json"))

    patch_param = []
    for patch in patches:
        patch_param.append({
        "parms":parms,
        "workdir":workdir,
        "patch":patch
    })

    patch_task(patch_param[0])
    # Run steps in parallel
    # # Run steps in parallel
    # with ThreadPool(processes=POOL_SIZE) as pool:
    #     pool.map(patch_task, patch_param)
        
    # step_5b_ps_merge_patches(workdir,patch,parms)
    # step_aps_linear(workdir)
    # step_6_ps_unwrap(workdir,parms)
    # step_7a_ps_calc_scla(workdir,parms)
    # step_7b_ps_smooth_scla(workdir)
