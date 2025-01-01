#!/usr/bin/env python3

import os
import sys
# import cupy as cp
import numpy as np
from pathlib import Path
from typing import List, Tuple, BinaryIO
import struct

def invalid_argc(argc: int) -> bool:
    """Check if the number of command line arguments is valid."""
    if argc < 3:
        print("Usage: selpsc parmfile patch.in pscands.1.ij pscands.1.da mean_amp.flt")
        print("input parameters:")
        print("  parmfile (input) amplitude dispersion threshold")
        print("                   width of amplitude files (range bins)")
        print("                   SLC file names & calibration constants")
        print("  patch.in (input) location of patch in rg and az")
        print("  pscands.1.ij   (output) PS candidate locations")
        print("  pscands.1.da   (output) PS candidate amplitude dispersion")
        print("  mean_amp.flt (output) mean amplitude of image")
        return True
    return False

def byteswap_complex(data: np.ndarray) -> np.ndarray:
    """Swap bytes for complex float data."""
    return np.array(data.byteswap())

def read_parmfile(filename: str) -> Tuple[float, int, List[Tuple[str, float]]]:
    """Read parameter file and return threshold, width and calibration factors."""
    with open(filename) as f:
        D_thresh = float(f.readline())
        width = int(f.readline())
        
        # Read calibration factors
        amp_files = []
        for line in f:
            if line.strip():
                fname, calib = line.strip().split()
                amp_files.append((fname, float(calib)))
                
    return D_thresh, width, amp_files

def read_patch_coords(filename: str) -> Tuple[int, int, int, int]:
    """Read patch coordinates."""
    with open(filename) as f:
        rg_start = int(f.readline())
        rg_end = int(f.readline())
        az_start = int(f.readline())
        az_end = int(f.readline())
    return rg_start, rg_end, az_start, az_end

def process_patch(amp_files: List[Tuple[str, float]], 
                 coords: Tuple[int, int, int, int],
                 width: int,
                 D_thresh: float,
                 output_files: Tuple[str, str, str, str, str]) -> None:
    """Process patch and identify PS candidates."""
    
    rg_start, rg_end, az_start, az_end = coords
    ijname, jiname, ijname0, daoutname, meanoutname = output_files
    
    # Calculate patch dimensions
    patch_width = rg_end - rg_start + 1
    patch_lines = az_end - az_start + 1
    
    # Initialize GPU arrays
    num_files = len(amp_files)
    patch_data = np.zeros((num_files, patch_lines, patch_width), dtype=np.complex64)
    calib_factors = np.array([f[1] for f in amp_files])
    
    # Read data into GPU arrays
    for i, (fname, _) in enumerate(amp_files):
        with open(fname, 'rb') as f:
            # Check for sun raster header
            header = f.read(32)
            if struct.unpack('>l', header[:4])[0] == 0x59a66a95:
                print("sun raster file - skipping header")
            else:
                f.seek(0)
                
            # Read patch data
            f.seek((az_start - 1) * width * 8 + (rg_start - 1) * 8)
            for line in range(patch_lines):
                data = np.fromfile(f, dtype=np.complex64, count=patch_width)
                patch_data[i, line] = np.array(data)
                f.seek((width - patch_width) * 8, 1)
    
    # Byteswap if needed
    patch_data = byteswap_complex(patch_data)
    
    # Calculate amplitudes
    amplitudes = np.abs(patch_data)
    
    # Calculate statistics
    sum_amp = np.sum(amplitudes / calib_factors[:, None, None], axis=0)
    sum_amp_sq = np.sum((amplitudes / calib_factors[:, None, None])**2, axis=0)
    
    # Calculate amplitude dispersion
    D_sq = num_files * sum_amp_sq / (sum_amp**2) - 1
    
    # Find PS candidates
    if D_thresh >= 0:
        ps_mask = (D_sq < D_thresh**2) & (sum_amp > 0)
    else:
        ps_mask = (D_sq >= D_thresh**2) & (sum_amp > 0)
    
    # Write results
    ps_mask_cpu = ps_mask
    D_sq_cpu = D_sq
    sum_amp_cpu = sum_amp
    
    pscid = 0
    with open(ijname, 'w') as fij, \
         open(jiname, 'wb') as fji, \
         open(ijname0, 'w') as fij0, \
         open(daoutname, 'w') as fda, \
         open(meanoutname, 'wb') as fmean:
        
        for y in range(patch_lines):
            for x in range(patch_width):
                if ps_mask_cpu[y, x]:
                    pscid += 1
                    az = az_start - 1 + y
                    rg = rg_start - 1 + x
                    
                    # Check for zero amplitudes
                    if np.any(amplitudes[:, y, x] <= 0.00005):
                        fij0.write(f"{pscid} {az} {rg}\n")
                    else:
                        fij.write(f"{pscid} {az} {rg}\n")
                        fji.write(struct.pack('>ii', rg, az))
                        fda.write(f"{np.sqrt(D_sq_cpu[y, x])}\n")
                
                # Write mean amplitude
                fmean.write(struct.pack('f', sum_amp_cpu[y, x]))

def run_pscpatch(patch_id:str,parmfile: str, patchfile: str, ijname: str, daoutname: str, meanoutname: str) -> None:
    print("Running pscpatch ...\t[{}]".format(patch_id))
    # Setup output filenames
    jiname = f"{ijname}.int"
    ijname0 = f"{ijname}0"
    
    # Read input files
    D_thresh, width, amp_files = read_parmfile(parmfile)
    coords = read_patch_coords(patchfile)
    
    # Process patch
    process_patch(amp_files, coords, width, D_thresh, 
                 (ijname, jiname, ijname0, daoutname, meanoutname))

def main():
    if invalid_argc(len(sys.argv)):
        return
        
    # Parse command line arguments
    parmfile = sys.argv[1]
    patchfile = sys.argv[2]
    ijname = sys.argv[3] if len(sys.argv) > 3 else "pscands.1.ij"
    daoutname = sys.argv[4] if len(sys.argv) > 4 else "pscands.1.da"
    meanoutname = sys.argv[5] if len(sys.argv) > 5 else "mean_amp.flt"
    
    run_pscpatch(parmfile, patchfile, ijname, daoutname, meanoutname)

# if __name__ == "__main__":
#     main()
