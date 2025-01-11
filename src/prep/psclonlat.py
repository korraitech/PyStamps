import h5py
import numpy
import torch
import struct
import os
from ..logger import appLogger
from ..misc import get_module_info

def run_psclonlat(patch_id: str, psclatlon_in: str, pscands_ij: str, pscands_ll: str):
    """
    Optimized version of run_psclonlat using PyTorch and CUDA (if available).
    The core calculations and output remain the same. 
    """
    
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {} {} {} {}".format(
        get_module_info(), patch_id, psclatlon_in, pscands_ij, pscands_ll
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Parse psclatlon_in to get width and .lonlat file names
    lonlat_files = []
    with open(psclatlon_in, "r") as parmfile:
        width = int(parmfile.readline().strip())
        lonlat_files.append(parmfile.readline().strip())
        lonlat_files.append(parmfile.readline().strip())

    # 2) Pre-load each .lonlat file fully into a PyTorch tensor on GPU (if available)
    ifg_tensors = []
    for lonlat_file_path in lonlat_files:
        file_size = os.path.getsize(lonlat_file_path)
        with open(lonlat_file_path, "rb") as f:
            header = f.read(32)
            # If the file does not have the special header at the start, rewind
            if struct.unpack(">l", header[:4])[0] != 0x59A66A95:
                # not the special marker, so revert
                f.seek(0)
                header_size = 0
            else:
                # if the header is valid, skip the 32 bytes
                header_size = 32

            # Compute how many bytes remain for float data
            data_size = file_size - header_size
            if data_size % 4 != 0:
                raise ValueError(f"File {lonlat_file_path} size is not divisible by 4 after header.")

            # Number of float32 elements
            num_elems = data_size // 4
            if num_elems % width != 0:
                raise ValueError(f"Width {width} does not evenly divide file array size.")

            height = num_elems // width
            # Read all float data
            raw_bin = f.read(data_size)
            # Convert to numpy array interpreting big-endian floats (>f4), then to torch tensor
            np_data = numpy.frombuffer(raw_bin, dtype=">f4").reshape(height, width)

            # Add these two lines to handle endianness:
            # Convert from big-endian to native byte order (little-endian on most systems).
            np_data = np_data.byteswap().newbyteorder()

            # Then convert to torch tensor
            ifg_tensors.append(torch.from_numpy(np_data).to(device))

    # 3) Read the PSC candidates from pscands_ij (HDF5)
    #    We expect a dataset called 'data' with shape (N, 3?) where [_, y, x]
    with h5py.File(pscands_ij, "r") as ij_hdf:
        ij_data = numpy.array(ij_hdf["data"])  # shape = (N, 3): (index, y, x) or similar
    # Move row/col indices to GPU
    y_coords = torch.from_numpy(ij_data[:, 1]).long().to(device)
    x_coords = torch.from_numpy(ij_data[:, 2]).long().to(device)

    # 4) Prepare the output dataset on CPU/HDF5
    num_points = len(ij_data)
    num_ifgs = len(ifg_tensors)

    # 5) Perform gather on GPU: each ifg_tensors[i] is shape (height, width).
    #    We gather the correct pixel for each (y, x).
    #    final_tensor will be (num_points, num_ifgs).
    final_tensor = torch.empty((num_points, num_ifgs), dtype=torch.float32, device=device)
    for ifg_index, ifg_tensor in enumerate(ifg_tensors):
        final_tensor[:, ifg_index] = ifg_tensor[y_coords, x_coords]

    # 6) Write the results to pscands_ll HDF5 (convert back to CPU float64 if desired).
    final_np = final_tensor.cpu().numpy().astype(numpy.float64)
    with h5py.File(pscands_ll, "w") as ll_hdf:
        ll_dataset = ll_hdf.create_dataset(
            "data",
            (num_points, num_ifgs),
            maxshape=(None, num_ifgs),
            dtype="d",
            chunks=True,
        )
        ll_dataset[:] = final_np