#########################################################################
#   Copyright 2025 - 2025, KorrAI                                       #
#   This program is free software: you can redistribute it and/or       #
#   modify it under the terms of the GNU General Public License as      #
#   published by the Free Software Foundation, either version 3 of      #
#   the License, or (at your option) any later version.                 #
#                                                                       #
#   This program is distributed in the hope that it will be useful,     #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of      #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the        #
#   GNU General Public License for more details.                        #
#                                                                       #
#   You should have received a copy of the GNU General Public License   #
#   along with this program. If not, see <https://www.gnu.org/licenses/>.#
#########################################################################
#                                                                       #
#   This file contains the caller of pystamps steps.                    #
#                                                                       #
#########################################################################


import os
import json
from .step.utils import read_lines
from .step.step_1_ps_loadgm import step_1_ps_loadgm
from .step.step_2_ps_estmgm import step_2_ps_estmgm
from .step.step_3_ps_select import step_3_ps_select
from .step.step_4_ps_weed import step_4_ps_weed
from .step.step_5_ps_merge import step_5_ps_merge
from .step.step_6_ps_unwrap import step_6_ps_unwrap
from .step.step_aps_linear import step_aps_linear
from .step.step_7_ps_scla import step_7_ps_scla
from .step.step_8_ps_plot import step_8_ps_plot
from multiprocessing import Pool

def read_json(path:str):
    data = {}
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def patch_task(parmas:dict):
    patch = parmas["patch"]
    parms = parmas["parms"]
    workdir = parmas["workdir"]
    step_1_ps_loadgm(workdir,patch)
    step_2_ps_estmgm(workdir,patch,parms)
    step_3_ps_select(workdir,patch,parms)
    step_4_ps_weed(workdir,patch,parms)

def run_stamps_steps(workdir:str,prg:int,paz:int):
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
    with Pool(processes=prg*paz) as pool:
        pool.map(patch_task, patch_param)
        
    step_5_ps_merge(workdir,parms)
    step_6_ps_unwrap(workdir,parms)
    step_aps_linear(workdir)
    step_7_ps_scla(workdir,parms)
    step_8_ps_plot(workdir,parms)
