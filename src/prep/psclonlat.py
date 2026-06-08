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
#   This file contains the implementation of psclonlat.                 #
#                                                                       #
#########################################################################


import h5py
import numpy as np
import struct
import os
from ..logger import appLogger
from ..misc import get_module_info

def run_psclonlat(patch_id: str, psclatlon_in: str, pscands_ij: str, pscands_ll: str, batch_size: int = 100000):
    """
    Optimized version of run_psclonlat using NumPy with batch processing to 
    accommodate large PSC lists when memory is limited.

    Args:
        patch_id: Identifier for the current patch.
        psclatlon_in: Path to parameter file containing width and two .lonlat filenames.
        pscands_ij: Path to HDF5 file containing PSC candidate row/column indices.
        pscands_ll: Output HDF5 file to store the gathered float values (lon/lat or similar).
        batch_size: Number of PSC points to process in each chunk (default = 100000).
    """
    
    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "Start"))

    # 1) Parse psclatlon_in to get width and .lonlat file names
    lonlat_files = []
    with open(psclatlon_in, "r") as parmfile:
        width = int(parmfile.readline().strip())
        lonlat_files.append(parmfile.readline().strip())
        lonlat_files.append(parmfile.readline().strip())

    # 2) Pre-load each .lonlat file fully into a NumPy array
    ifg_arrays = []
    for lonlat_file_path in lonlat_files:
        file_size = os.path.getsize(lonlat_file_path)
        with open(lonlat_file_path, "rb") as f:
            header = f.read(32)
            # Check special header
            if struct.unpack(">l", header[:4])[0] != 0x59A66A95:
                # not the special marker, revert
                f.seek(0)
                header_size = 0
            else:
                # valid header, skip the 32 bytes
                header_size = 32

            # Verify that remaining data is a multiple of 4
            data_size = file_size - header_size
            if data_size % 4 != 0:
                raise ValueError(f"File {lonlat_file_path} size is not divisible by 4 after header.")

            num_elems = data_size // 4
            if num_elems % width != 0:
                raise ValueError(f"Width {width} does not evenly divide file array size.")

            height = num_elems // width
            raw_bin = f.read(data_size)
            # Interpret big-endian float32, then swap to native (little-endian)
            np_data = np.frombuffer(raw_bin, dtype=">f4").reshape(height, width)
            np_data = np_data.byteswap().view(np_data.dtype.newbyteorder())

            # Store as NumPy array
            ifg_arrays.append(np_data)

    # 3) Read the PSC candidates from pscands_ij (HDF5)
    #    We expect a dataset called 'data' with shape (N, 3): (index, y, x)
    with h5py.File(pscands_ij, "r") as ij_hdf:
        ij_data = np.array(ij_hdf["data"])  # shape = (N, 3)

    # Extract row/col indices
    y_coords = ij_data[:, 1].astype(np.int64)
    x_coords = ij_data[:, 2].astype(np.int64)

    num_points = len(ij_data)
    num_ifgs = len(ifg_arrays)

    # 4) Create the output HDF5 dataset with space for all points/ifgs.
    #    We'll fill it in batches to manage memory usage.
    with h5py.File(pscands_ll, "w") as ll_hdf:
        ll_dataset = ll_hdf.create_dataset(
            "data",
            (num_points, num_ifgs),
            maxshape=(None, num_ifgs),
            dtype="d",
            chunks=True,
        )
        
        # 5) Batch processing over PSC points.
        for start_idx in range(0, num_points, batch_size):
            end_idx = min(start_idx + batch_size, num_points)
            batch_size_actual = end_idx - start_idx

            # Slice the row/col for this batch
            batch_y = y_coords[start_idx:end_idx]
            batch_x = x_coords[start_idx:end_idx]

            # Create batch result array
            batch_result = np.empty((batch_size_actual, num_ifgs), dtype=np.float32)

            # Gather for each ifg array
            for ifg_index, ifg_array in enumerate(ifg_arrays):
                batch_result[:, ifg_index] = ifg_array[batch_y, batch_x]

            # Convert to float64 for writing
            batch_np = batch_result.astype(np.float64)
            ll_dataset[start_idx:end_idx, :] = batch_np

    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "End"))
