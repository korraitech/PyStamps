#########################################################################
#   Copyright 2025 - 2025, KorrAI                                       #
#   This program is free software: you can redistribute it and/or       #
#   modify it under the terms of the European Space Agency Public       #
#   License (ESA-PL) Permissive (Type 3) - v2.4.                        #
#                                                                       #
#   This program is distributed in the hope that it will be useful,     #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of      #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the        #
#   ESA-PL Permissive (Type 3) - v2.4 for more details.                 #
#                                                                       #
#   You should have received a copy of the license along with this      #
#   program. If not, see the ESA-PL v2.4 license at:                    #
#   https://essr.esa.int/license/european-space-agency-public-license-v2-4-permissive-type-3
#########################################################################
#                                                                       #
#   This file contains the implementation of pscdem.                    #
#                                                                       #
#########################################################################


import h5py
from array import array
from ..logger import appLogger
from ..misc import get_module_info

def run_pscdem(patch_id: str, pscphase_in: str, pscands_ij: str, pscands_ht: str, chunk_height: int=5000):
    """
    Optimized version of run_pscdem that reads and processes the interferogram in batches.

    :param patch_id: String identifying the current patch.
    :param pscphase_in: Path to the pscphase input file (contains width and ifg filename).
    :param pscands_ij: Path to the HDF5 of PSC candidate row/col indices.
    :param pscands_ht: Path to the output HDF5 file for writing heights.
    :param chunk_height: How many rows to read at a time from the interferogram file.
    """
    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "Start"))

    # Read parameters from pscphase_in
    with open(pscphase_in, 'r') as parmfile:
        width = int(parmfile.readline().strip())
        ifgfilename = parmfile.readline().strip()

    # Open the interferogram binary file
    ifgfile = open(ifgfilename, 'rb')

    # Read and possibly skip the 32-byte header if it indicates a certain format
    header = ifgfile.read(32)
    if int.from_bytes(header[:4], 'little') == 0x59a66a95:
        print("pscdem: sun raster file - skipping header")
    else:
        # If it's not that header signature, seek back to the start
        ifgfile.seek(0)

    # Figure out total size in floats (after accounting for any skipped header)
    file_size_bytes = 0
    ifgfile.seek(0, 2)  # move to end
    file_size_bytes = ifgfile.tell()
    ifgfile.seek(0, 0)  # move back to start

    # If we skipped the header, adjust the file size
    # The code above might have read 32 bytes—if it was recognized, we keep it consumed
    # If not recognized, we rewound, so no adjustment needed
    recognized_header = (int.from_bytes(header[:4], 'little') == 0x59a66a95)
    if recognized_header:
        # we've already consumed 32 bytes
        data_start_offset = 32
    else:
        data_start_offset = 0

    total_bytes = file_size_bytes - data_start_offset
    total_floats = total_bytes // 4
    height = total_floats // width

    # Move file pointer to the actual start of float data
    ifgfile.seek(data_start_offset, 0)

    # Read PSC candidate (row,col) data
    with h5py.File(pscands_ij, 'r') as ij_hdf:
        ij_array = ij_hdf['data'][:]  # Read all data into memory
    
    # Extract the y and x coordinates 
    y_coords = []
    x_coords = []
    for i in range(len(ij_array)):
        y_coords.append(int(ij_array[i][1]))
        x_coords.append(int(ij_array[i][2]))
    num_candidates = len(y_coords)

    # Create the output HDF5 and dataset
    with h5py.File(pscands_ht, 'w') as ht_hdf:
        # We'll create a dataset of shape (num_candidates,) and store float results here
        ht_dset = ht_hdf.create_dataset(
            'data',
            (num_candidates,),
            maxshape=(None,),
            dtype='f',
            chunks=True
        )

        # We'll process chunks of the interferogram file
        current_row_start = 0
        while current_row_start < height:
            current_row_end = min(current_row_start + chunk_height, height)
            rows_to_read = current_row_end - current_row_start

            # How many floats in this chunk?
            floats_in_chunk = rows_to_read * width

            # Read exactly the bytes for this chunk
            chunk_buffer = ifgfile.read(floats_in_chunk * 4)
            if not chunk_buffer:
                break  # no more data to read

            # Convert buffer to float32 using array module
            chunk_data = array('f')
            chunk_data.frombytes(chunk_buffer)
            # Convert big-endian to native byte order if needed
            chunk_data.byteswap()  # IEEE 754 floats are big-endian in the file

            # Find indices of PSCs in this chunk
            chunk_indices = []
            local_positions = []
            
            for i in range(num_candidates):
                if current_row_start <= y_coords[i] < current_row_end:
                    chunk_indices.append(i)
                    # Calculate local position in the chunk
                    local_y = y_coords[i] - current_row_start
                    local_x = x_coords[i]
                    local_pos = (local_y * width) + local_x
                    local_positions.append(local_pos)
            
            # Extract values for PSCs in this chunk
            results = []
            for pos in local_positions:
                if 0 <= pos < len(chunk_data):
                    results.append(chunk_data[pos])
                else:
                    results.append(0.0)  # Default value if position is out of bounds
            
            # Write back to HDF5 dataset at the correct PSC candidate positions
            for idx, result_idx in enumerate(chunk_indices):
                ht_dset[result_idx] = results[idx]

            current_row_start = current_row_end

    ifgfile.close()
    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "End"))
