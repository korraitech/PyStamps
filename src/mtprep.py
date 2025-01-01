#!/usr/bin/env python
# Interface between Snap and StaMPS
#
# Andy Hooper, August 2017
# Rahul V Sharan, February 2024
# Rahul V Sharan, December 2024
# ===========================================================================
# Updated code to parallelize cands using python script
# ===========================================================================
# Expected directory structure:
# PS:
#   rslc/*.rslc
#   rslc/*.rslc.par
#   diff0/*.diff
#   diff0/*.base
#   geo/*dem.rdc
#   geo/*diff_par
#   geo/YYYYMMDD.lon (master)
#   geo/YYYYMMDD.lat (master)
#   dem/*_seg.par
# ===========================================================================
import os
from .logger import appLogger
from .prep.calamp import run_calamp
from .prep.pscpatch import run_pscpatch
from .prep.psclonlat import run_psclonlat
from .prep.pscdem import run_pscdem
from .prep.pscphase import run_pscphase
from .prep.parminit import ps_parms_init
from .env import CONCURRENT
from multiprocessing import Pool

def parse_field(path:str):
    field_data = {}
    with open(path) as fp:
        for line in fp.readlines():
            line_data = line.strip().split(':')
            field_data[line_data[0]] = line_data[1].replace('\t','#')
    return field_data

def find_file(datadir:str,subdir:str,filename:str):
    return os.path.join(datadir,f'{subdir}',f'{filename}')

def find_files(datadir:str,subdir:str,pattern:str,place:str = ("","")):
    paths = []
    rslc_path = os.path.join(datadir,subdir)
    for file_name in os.listdir(rslc_path):
        if pattern not in file_name:
            paths.append(os.path.join(rslc_path,file_name).replace(place[0],place[1]))
    paths.sort()
    return paths

def save_meta(path:str,data:list):
    with open(path, "w") as f:
        for line in data:
            f.write(str(line) + '\n')

def read_meta(path:str):
    meta = []
    with open(path) as f:
        for line in f.readlines():
            meta.append(line.strip())
    return meta

def create_patch(workdir,length,width,prg,paz,overlap_rg,overlap_az):
    appLogger.info("MT PREP :: PATCH :: {} {} {} {} {} {} {}".format(
        workdir,length,width,prg,paz,overlap_rg,overlap_az))
    length_p = int(length / paz)
    width_p = int(width / prg)
    irg = 0
    ip = 0
    patch_list = []
    while irg < prg:
        irg += 1
        iaz = 0
        while iaz < paz:
            iaz += 1
            ip += 1
            start_rg1 = width_p * (irg - 1) + 1
            start_rg = 1  if start_rg1 - overlap_rg < 1 else start_rg1 - overlap_rg
            end_rg1 = width_p * irg
            end_rg = end_rg1 + overlap_rg if end_rg1 + overlap_rg <= width else width
            start_az1 = length_p * (iaz - 1) + 1
            start_az = 1 if start_az1 - overlap_az < 1 else start_az1 - overlap_az
            end_az1 = length_p * iaz
            end_az = end_az1 + overlap_az if end_az1 + overlap_az <= length else length

            patch_list.append(f"PATCH_{ip}")
            os.makedirs(os.path.join(workdir,f"PATCH_{ip}"),exist_ok=True)
            save_meta(os.path.join(workdir,f"PATCH_{ip}", "patch.in"),[start_rg,end_rg,start_az,end_az])
            save_meta(os.path.join(workdir,f"PATCH_{ip}", "patch_noover.in"),[start_rg1,end_rg1,start_az1,end_az1])
    save_meta(os.path.join(workdir, "patch.list"),patch_list)
    return patch_list

def task_cands(params):
    workdir = params["workdir"]
    patchdir = os.path.join(workdir,params["patch"])
    pscands_1_ij = os.path.join(patchdir,'pscands.1.ij')
    
    run_pscpatch(params["patch"],
        os.path.join(workdir,'selpsc.in'),
        os.path.join(patchdir,'patch.in'),
        pscands_1_ij,os.path.join(patchdir,'pscands.1.da'),
        os.path.join(patchdir,'mean_amp.flt'))

    run_psclonlat(params["patch"],
        os.path.join(workdir,'psclonlat.in'),
        pscands_1_ij,os.path.join(patchdir,'pscands.1.ll'))
    
    run_pscdem(params["patch"],
        os.path.join(workdir,'pscdem.in'),
        pscands_1_ij,os.path.join(patchdir,'pscands.1.hgt'))
    
    run_pscphase(params["patch"],
        os.path.join(workdir,'pscphase.in'),
        pscands_1_ij,os.path.join(patchdir,'pscands.1.ph'))

def mt_prep(master:str,datadir:str,workdir:str,da_thresh:float=0.4,
              prg:int=2, paz:int=2, overlap_rg:int= 50,overlap_az:int =200):
    appLogger.info("MT PREP :: {} {} {} {} {} {} {} {}".format(
        master,datadir,workdir,da_thresh,prg,paz,overlap_rg,overlap_az))
    
    rsc = find_file(datadir,"rslc",f"{master}.rslc.par")
    
    rsc_data = parse_field(rsc)
    length = int(rsc_data["azimuth_lines"].replace('#',''))
    width = int(rsc_data["range_samples"].replace('#',''))

    save_meta(os.path.join(workdir, "processor.txt"),["snap"])
    save_meta(os.path.join(workdir, "width.txt"),[width])
    save_meta(os.path.join(workdir, "len.txt"),[length])
    save_meta(os.path.join(workdir, "rsc.txt"),[rsc])

    # Calibrate amplitudes
    calamp_in = os.path.join(workdir, "calamp.in")
    calamp_out = os.path.join(workdir, "calamp.out")
    save_meta(calamp_in,find_files(datadir,'rslc','.rslc.par'))
    run_calamp(calamp_in=calamp_in,width=width, calamp_out=calamp_out)
    save_meta(os.path.join(workdir, "selpsc.in"),[da_thresh,width]+read_meta(calamp_out))
    
    # Create Patches
    patch_list = create_patch(workdir,length,width,prg,paz,overlap_rg,overlap_az)
    save_meta(os.path.join(workdir, "pscdem.in"),[width,find_file(datadir,"geo","elevation_dem.rdc")])
    save_meta(os.path.join(workdir, "pscphase.in"),[width]+find_files(datadir,'diff0','.diff',('base','diff')))
    save_meta(os.path.join(workdir, "psclonlat.in"),
              [width,find_file(datadir,"geo",f"{master}.lon"),find_file(datadir,"geo",f"{master}.lat")])
    
    # Process Patches
    task_param = []
    for patch in patch_list:
        task_param.append({"workdir": workdir,"patch": patch})
    
    with Pool(processes=CONCURRENT) as pool:
        pool.map(task_cands, task_param)    

    rslc_par = {
        "heading" : float(rsc_data["heading"].replace('#','').replace('degrees','')),
        "lambda" : 299792458/float(rsc_data["radar_frequency"].replace('#','').replace('Hz',''))
    }
    ps_parms_init(workdir=workdir, rslc_par=rslc_par)
