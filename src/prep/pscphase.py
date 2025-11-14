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
#   This file contains the implementation of pscphase.                  #
#                                                                       #
#########################################################################

import h5py
import numpy as np
import struct
from ..logger import appLogger
from ..misc import get_module_info

def run_pscphase(patch_id: str, pscphase_in: str, pscands_ij: str, pscands_ph: str, batch_size: int = 10000):
    """
    Optimized version of run_pscphase using NumPy for vectorized data loading, 
    but with batch processing to handle low-memory environments.

    Reads interferogram filenames, reads each file once into a float array,
    then processes a limited batch of indices at a time to avoid large memory usage.
    """
    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "Start"))

    # 1) Read widths and ifg_filenames:
    with open(pscphase_in, "r") as parmfile:
        width = int(parmfile.readline().strip())
        ifg_filenames = parmfile.read().splitlines()

    # 2) Read the (i, y, x) data from pscands_ij:
    with h5py.File(pscands_ij, "r") as ij_hdf:
        ij_data = ij_hdf["data"][:]  # shape [num_points, 3], each row presumably (i, y, x)
        y_coords = ij_data[:, 1]
        x_coords = ij_data[:, 2]
        num_points = len(ij_data)

    # 3) Create the output phase dataset in pscands_ph:
    with h5py.File(pscands_ph, "w") as ph_hdf:
        ph_dataset = ph_hdf.create_dataset(
            "data",
            (num_points, len(ifg_filenames)),
            maxshape=(None, len(ifg_filenames)),
            dtype=np.complex64,
            chunks=True,
        )

        # Use NumPy arrays for coordinates
        y_coords = y_coords.astype(np.int64)
        x_coords = x_coords.astype(np.int64)

        # 4) Loop over each ifg_filename but process in batches to limit memory usage:
        for ifg_index, ifg_filename in enumerate(ifg_filenames):
            with open(ifg_filename, "rb") as ifgfile:
                header = ifgfile.read(32)
                # Check for signature; if not found, reset to start of file:
                if struct.unpack(">l", header[:4])[0] != 0x59A66A95:
                    ifgfile.seek(0)

                # Read remainder of file as big-endian float32 data, then reshape to Nx2 for real/imag:
                file_data = np.fromfile(ifgfile, dtype=">f4")  # big-endian float32
                # Convert big-endian to native (little-endian on most machines)
                file_data = file_data.astype("<f4")

            # Reshape into shape [num_pixels, 2]
            num_pixels = file_data.size // 2
            file_data = file_data.reshape(num_pixels, 2)

            # Pre-compute the absolute indices once for all points
            # index = y*width + x
            index = y_coords * width + x_coords

            # 5) Process these extracted pixel arrays in manageable batches:
            start = 0
            while start < num_points:
                end = min(start + batch_size, num_points)
                sub_idx = index[start:end]

                # Gather from the main data using NumPy advanced indexing
                gathered = file_data[sub_idx]

                # Convert to a complex64 array for writing
                real_part = gathered[:, 0]
                imag_part = gathered[:, 1]
                temp_array = (real_part + 1j * imag_part).astype(np.complex64)

                # Write the extracted pixel array into the HDF5 dataset for this batch
                ph_dataset[start:end, ifg_index] = temp_array

                start = end

    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "End"))
