import h5py
import numpy
import struct
import torch
from ..logger import appLogger
from ..misc import get_module_info

def run_pscphase(patch_id: str, pscphase_in: str, pscands_ij: str, pscands_ph: str):
    """
    Optimized version of run_pscphase using PyTorch for vectorized data loading.

    Reads ifg filenames, reads each file once into a float array, uses advanced indexing
    to gather the appropriate pixel values, and writes out the complex data to pscands_ph.
    """
    appLogger.info(
        ">>>>>>>>>>>>>>>> {}\t\t|| {} {} {} {}".format(
            get_module_info(), patch_id, pscphase_in, pscands_ij, pscands_ph
        )
    )

    # 1) Read widths and ifg_filenames:
    with open(pscphase_in, "r") as parmfile:
        width = int(parmfile.readline().strip())
        ifg_filenames = parmfile.read().splitlines()

    # 2) Read the (i, y, x) data from pscands_ij:
    with h5py.File(pscands_ij, "r") as ij_hdf:
        ij_data = ij_hdf["data"][:]  # shape [num_points, 3], each row presumably (i, y, x)
        # If your dataset is [ (i, y, x), (i, y, x), ... ], collect y & x columns to use for indexing.
        y_coords = ij_data[:, 1]
        x_coords = ij_data[:, 2]
        num_points = len(ij_data)

    # 3) Create the output phase dataset in pscands_ph:
    with h5py.File(pscands_ph, "w") as ph_hdf:
        ph_dataset = ph_hdf.create_dataset(
            "data",
            (num_points, len(ifg_filenames)),
            maxshape=(None, len(ifg_filenames)),
            dtype=numpy.complex64,
            chunks=True,
        )

        # Convert x_coords and y_coords into Torch tensors (on CPU by default).
        # You could place them on GPU if you wanted to do further GPU-based processing.
        y_t = torch.from_numpy(y_coords.astype(numpy.int64))
        x_t = torch.from_numpy(x_coords.astype(numpy.int64))
        width_t = torch.tensor(width, dtype=torch.int64)

        # 4) Loop over each ifg_filename but try to vectorize the pixel extraction:
        for i, ifg_filename in enumerate(ifg_filenames):
            with open(ifg_filename, "rb") as ifgfile:
                header = ifgfile.read(32)
                # Check for signature; if not found, reset to start of file:
                if struct.unpack(">l", header[:4])[0] != 0x59A66A95:
                    ifgfile.seek(0)

                # Read remainder of file as big-endian float32 data, then reshape to Nx2 for real/imag:
                # Each pixel is 8 bytes: float32 (real) + float32 (imag).
                file_data = numpy.fromfile(ifgfile, dtype=">f4")  # big-endian float32
                # Convert big-endian to native (little-endian on most machines)
                file_data = file_data.astype("<f4")

                # If we know how many rows total, or have the total array size, we can reshape:
                # For example, if total length in floats is 2*(height*width), you can do:
                # file_data = file_data.reshape(-1, 2)
                # However, if the actual dimension is unknown, you might do a check before.
                # For demonstration, assume we can do the following:
                num_pixels = file_data.size // 2
                file_data = file_data.reshape(num_pixels, 2)

                # Convert to Torch on CPU (or GPU if needed)
                file_data_t = torch.from_numpy(file_data)  # now in native order

                #  (row-major indexing) gather all the relevant pixel positions
                # index_t = y*width + x
                index_t = y_t * width_t + x_t

                # index into file_data_t with advanced indexing:
                # We want file_data_t[index_t, :] which will be shape [num_points, 2]
                gathered = file_data_t[index_t]

                # Convert that to a complex64 array for writing.
                # Torch doesn't have a built-in complex64 container for gather, so we manually stack:
                real_part = gathered[:, 0]
                imag_part = gathered[:, 1]
                # Move back to numpy complex64:
                temp_array = (real_part.numpy() + 1j * imag_part.numpy()).astype(numpy.complex64)

            # 5) Write the extracted pixel array into the HDF5 dataset:
            ph_dataset[:, i] = temp_array
