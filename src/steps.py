import os
import json
from .step.utils import read_lines
from .step.ps_load_gamma import ps_load_gamma
from .step.ps_estm_gamma import ps_estm_gamma
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
    ps_load_gamma(workdir,patch)
    ps_estm_gamma(workdir,patch,parms)

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
    with ThreadPool(processes=4) as pool:
        pool.map(patch_task, patch_param)
