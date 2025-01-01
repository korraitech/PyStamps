#!/usr/bin/env python3

import sys
#import cupy as cp
import numpy as np
from typing import List, Tuple
import struct

def invalid_argc(argc: int) -> bool:
    """Check if the number of command line arguments is valid."""
    if argc < 2:
        print("Usage: pscphase parmfile pscands.1.ij pscands.1.ph\n")
        print("Input parameters:")
        print("  parmfile   (input)  width of interferogram files (range bins)")
        print("                      list of interferogram files (complex float)")
        print("  pscands.1.ij (input)  location of permanent scatterer candidates")
        print("  pscands.1.ph (output) phase of permanent scatterer candidates")
        return True
    return False

def read_parmfile(filename: str) -> Tuple[int, List[str]]:
    """Read parameter file and return width and interferogram filenames."""
    ifg_files = []
    with open(filename) as f:
        width = int(f.readline().strip())
        for line in f:
            if line.strip():
                ifg_files.append(line.strip())
    return width, ifg_files

def read_ps_locations(filename: str) -> List[Tuple[int, int, int]]:
    """Read PS candidate locations."""
    locations = []
    with open(filename) as f:
        for line in f:
            if line.strip():
                pscid, y, x = map(int, line.strip().split())
                locations.append((pscid, y, x))
    return locations

def pscphase_utils(ifg_files: List[str], 
                         width: int,
                         ps_locations: List[Tuple[int, int, int]],
                         outfile: str) -> None:
    """Process interferograms and extract phase for PS candidates."""
    
    # Constants
    HEADER_SIZE = 32
    MAGIC_NUMBER = 0x59a66a95
    COMPLEX_SIZE = 8  # size of complex float (2 * 4 bytes)
    
    num_files = len(ifg_files)
    num_ps = len(ps_locations)
    
    # Preallocate output array on GPU
    phase_data = np.zeros((num_ps, num_files), dtype=np.complex64)
    
    # Process each interferogram
    for i, ifg_file in enumerate(ifg_files):
        
        with open(ifg_file, 'rb') as f:
            # Check for sun raster header
            header = f.read(HEADER_SIZE)
            if struct.unpack('>L', header[:4])[0] == MAGIC_NUMBER:
                print("sun raster file - skipping header")
            else:
                f.seek(0)
            
            # Read phase data for each PS location
            for j, (pscid, y, x) in enumerate(ps_locations):
                # Calculate file position
                pos = (y * width + x) * COMPLEX_SIZE
                f.seek(pos)
                
                # Read complex value
                real, imag = struct.unpack('ff', f.read(COMPLEX_SIZE))
                phase_data[j, i] = complex(real, imag)
    
    # Write output
    with open(outfile, 'wb') as f:
        # Transfer data back to CPU and write
        for ps_idx in range(num_ps):
            for ifg_idx in range(num_files):
                complex_val = phase_data[ps_idx, ifg_idx]
                f.write(struct.pack('ff', complex_val.real, complex_val.imag))

def run_pscphase(patch_id:str, parmfile: str, ijname: str, outfile: str) -> None:
    print("Running pscphase ...\t[{}]".format(patch_id))

    # Read input files
    width, ifg_files = read_parmfile(parmfile)
    ps_locations = read_ps_locations(ijname)
    
    # Process interferograms
    pscphase_utils(ifg_files, width, ps_locations, outfile)

def main():
    if invalid_argc(len(sys.argv)):
        return
        
    # Parse command line arguments
    parmfile = sys.argv[1]
    ijname = sys.argv[2] if len(sys.argv) > 2 else "pscands.1.ij"
    outfile = sys.argv[3] if len(sys.argv) > 3 else "pscands.1.ph"
    
    run_pscphase("patch_id",parmfile, ijname, outfile)

# if __name__ == "__main__":
#     main()
