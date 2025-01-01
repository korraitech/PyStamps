#!/usr/bin/env python3

import sys
#import cupy as cp
import numpy as np
from typing import List, Tuple
import struct

def invalid_argc(argc: int) -> bool:
    """Check if the number of command line arguments is valid."""
    if argc < 2:
        print("Usage: psclonlat parmfile pscands.1.ij pscands.1.ll\n")
        print("Input parameters:")
        print("  parmfile   (input)  width of lon/lat files (range bins)")
        print("                      name of lon file (float)")
        print("                      name of lat file (float)")
        print("  pscands.1.ij (input)  location of permanent scatterer candidates")
        print("  pscands.1.ll (output) lon/lat of permanent scatterer candidates\n")
        return True
    return False

def read_parmfile(filename: str) -> Tuple[int, List[str]]:
    """Read parameter file and return width and lon/lat filenames."""
    with open(filename) as f:
        width = int(f.readline().strip())
        # Read lon and lat filenames
        lon_file = f.readline().strip()
        lat_file = f.readline().strip()
    return width, [lon_file, lat_file]

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

def psclonlat_utils(coord_files: List[str], 
                  width: int,
                  ps_locations: List[Tuple[int, int, int]],
                  outfile: str) -> None:
    """Process longitude and latitude files for PS candidates."""
    
    # Constants
    HEADER_SIZE = 32
    MAGIC_NUMBER = 0x59a66a95
    FLOAT_SIZE = 4
    
    num_ps = len(ps_locations)
    
    # Allocate GPU arrays for coordinates
    coords = np.zeros((num_ps, 2), dtype=np.float32)
    
    # Process lon and lat files
    for i, filename in enumerate(coord_files):
        with open(filename, 'rb') as f:
            # Check for sun raster header
            header = f.read(HEADER_SIZE)
            if struct.unpack('>L', header[:4])[0] == MAGIC_NUMBER:
                print("sun raster file - skipping header")
            else:
                f.seek(0)
            
            # Read coordinates for each PS location
            for j, (pscid, y, x) in enumerate(ps_locations):
                # Calculate file position
                pos = (y * width + x) * FLOAT_SIZE
                f.seek(pos)
                
                # Read coordinate value
                coord = struct.unpack('f', f.read(FLOAT_SIZE))[0]
                coords[j, i] = coord
                
                # Progress reporting
                if pscid % 100000 == 0:
                    print(f"{pscid} PS candidates processed")
    
    # # Write output
    with open(outfile, 'wb') as f:
        # Transfer data back to CPU and write
        for ps_coords in coords:
            # Write lon/lat pair
            f.write(struct.pack('ff', ps_coords[0], ps_coords[1]))

# Large DataSet
def process_lonlat_batched(coord_files: List[str], 
                          width: int,
                          ps_locations: List[Tuple[int, int, int]],
                          outfile: str,
                          batch_size: int = 1000) -> None:
    """Process longitude and latitude files in batches."""
    
    HEADER_SIZE = 32
    MAGIC_NUMBER = 0x59a66a95
    FLOAT_SIZE = 4
    
    num_ps = len(ps_locations)
    
    with open(outfile, 'wb') as fout:
        # Process in batches
        for start_idx in range(0, num_ps, batch_size):
            end_idx = min(start_idx + batch_size, num_ps)
            batch_locations = ps_locations[start_idx:end_idx]
            
            # Allocate batch array on GPU
            coords = np.zeros((len(batch_locations), 2), dtype=np.float32)
            
            # Process lon and lat files
            for i, filename in enumerate(coord_files):
                with open(filename, 'rb') as fin:
                    # Check header
                    header = fin.read(HEADER_SIZE)
                    if struct.unpack('>L', header[:4])[0] == MAGIC_NUMBER:
                        print("sun raster file - skipping header")
                    else:
                        fin.seek(0)
                    
                    # Process batch
                    for j, (pscid, y, x) in enumerate(batch_locations):
                        pos = (y * width + x) * FLOAT_SIZE
                        fin.seek(pos)
                        coord = struct.unpack('f', fin.read(FLOAT_SIZE))[0]
                        coords[j, i] = coord
                        
                        if pscid % 100000 == 0:
                            print(f"{pscid} PS candidates processed")
            
            # Write batch results
            coords.tofile(fout)

def run_psclonlat(patch_id:str, parmfile:str, ijname:str, outfile:str) -> None:
    print("Running psclonlat ...\t[{}]".format(patch_id))
    
    # Read input files
    width, coord_files = read_parmfile(parmfile)
    ps_locations = read_ps_locations(ijname)
    
    # Process longitude and latitude
    psclonlat_utils(coord_files, width, ps_locations, outfile)

def main():
    if invalid_argc(len(sys.argv)):
        return 1
        
    # Parse command line arguments
    parmfile = sys.argv[1]
    ijname = sys.argv[2] if len(sys.argv) > 2 else "pscands.1.ij"
    outfile = sys.argv[3] if len(sys.argv) > 3 else "pscands.1.ll"
    
    run_psclonlat("patch_id",parmfile, ijname, outfile)

# if __name__ == "__main__":
#     main()
