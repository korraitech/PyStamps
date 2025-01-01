#!/usr/bin/env python3

import os
import sys
# import cupy as cp
import numpy as np
from typing import List, Tuple
import struct

def invalid_argc(argc: int) -> bool:
    """Check if the number of command line arguments is valid."""
    if argc < 2:
        print("Usage: pscdem pscands.1.ij pscands.1.hgt precision")
        print("Input parameters:")
        print("  parmfile   (input)  width of dem files (range bins)")
        print("                      name of dem file (radar coords, float)")
        print("  pscands.1.ij (input)  location of PS candidates")
        print("  pscands.1.hgt (output) height of PS candidates")
        return True
    return False

def read_parmfile(filename: str) -> Tuple[int, str]:
    """Read parameter file and return width and DEM filename."""
    with open(filename) as f:
        width = int(f.readline().strip())
        dem_file = f.readline().strip()
    return width, dem_file

def read_ps_locations(filename: str) -> List[Tuple[int, int, int]]:
    """Read PS candidate locations."""
    locations = []
    with open(filename) as f:
        for line in f:
            if line.strip():
                try:
                    pscid, y, x = map(int, line.strip().split())
                    locations.append((pscid, y, x))
                except ValueError:
                    continue
    return locations

def pscdem_utils(dem_file: str, 
               width: int,
               ps_locations: List[Tuple[int, int, int]],
               outfile: str) -> None:
    """Process DEM and extract heights for PS candidates."""
    
    # Constants
    HEADER_SIZE = 32
    MAGIC_NUMBER = 0x59a66a95
    FLOAT_SIZE = 4
    
    # Check if DEM file exists
    nodem_sw = not os.path.exists(dem_file)
    
    # Prepare GPU arrays
    num_ps = len(ps_locations)
    heights = np.zeros(num_ps, dtype=np.float32)
    
    with open(dem_file, 'rb') as f:
        # Check for sun raster header
        header = f.read(HEADER_SIZE)
        if struct.unpack('>L', header[:4])[0] == MAGIC_NUMBER:
            print("pscdem: sun raster file - skipping header")
        else:
            f.seek(0)
        
        # Read heights for each PS location
        for i, (pscid, y, x) in enumerate(ps_locations):
            # Calculate file position
            pos = (y * width + x) * FLOAT_SIZE
            f.seek(pos)
            
            # Read height value
            height = struct.unpack('f', f.read(FLOAT_SIZE))[0]
            heights[i] = height
            
            # Progress reporting
            if pscid % 100000 == 0:
                print(f"pscdem: {pscid} PS candidates processed")
    
    # Write output
    with open(outfile, 'wb') as f:
        # Transfer data back to CPU and write
        for height in heights:
            f.write(struct.pack('f', height))

def run_pscdem(patch_id:str, parmfile: str, ijname: str, outfile: str) -> None:
    print("Running pscdem ...\t[{}]".format(patch_id))

    # Read input files
    width, dem_file = read_parmfile(parmfile)
    ps_locations = read_ps_locations(ijname)
    
    # Process DEM
    pscdem_utils(dem_file, width, ps_locations, outfile)

def main():
    if invalid_argc(len(sys.argv)):
        return
        
    # Parse command line arguments
    parmfile = sys.argv[1]
    ijname = sys.argv[2] if len(sys.argv) > 2 else "pscands.1.ij"
    outfile = sys.argv[3] if len(sys.argv) > 3 else "pscands.1.hgt"

    run_pscdem("patch_id",parmfile, ijname, outfile)

# if __name__ == "__main__":
#     main()
