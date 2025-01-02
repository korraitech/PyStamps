import os
import json
from .step.utils import read_lines
from .step.step_1_ps_load_gamma import step_1_ps_load_gamma
from .step.step_2_ps_estm_gamma import step_2_ps_estm_gamma
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

    # Run steps in parallel
    POOL_SIZE = 1
    with ThreadPool(processes=POOL_SIZE) as pool:
        pool.map(patch_task, patch_param)
    
    # patch_task(patch_param[0])