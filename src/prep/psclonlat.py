import h5py
import numpy
import torch
import struct
import os
from ..logger import appLogger
from ..misc import get_module_info

def run_psclonlat(patch_id: str, psclatlon_in: str, pscands_ij: str, pscands_ll: str, batch_size: int = 100000):
    """
    Optimized version of run_psclonlat using PyTorch and CUDA (if available), 
    adding batch processing to accommodate large PSC lists when GPU memory is low.

    Args:
        patch_id: Identifier for the current patch.
        psclatlon_in: Path to parameter file containing width and two .lonlat filenames.
        pscands_ij: Path to HDF5 file containing PSC candidate row/column indices.
        pscands_ll: Output HDF5 file to store the gathered float values (lon/lat or similar).
        batch_size: Number of PSC points to process in each chunk (default = 100000).
    """
    
    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "Start"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Parse psclatlon_in to get width and .lonlat file names
    lonlat_files = []
    with open(psclatlon_in, "r") as parmfile:
        width = int(parmfile.readline().strip())
        lonlat_files.append(parmfile.readline().strip())
        lonlat_files.append(parmfile.readline().strip())

    # 2) Pre-load each .lonlat file fully into a PyTorch tensor (CPU or GPU).
    #    If the .lonlat files are very large, consider storing these on CPU instead 
    #    and only moving partial data or performing CPU gather below.
    ifg_tensors = []
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
            np_data = numpy.frombuffer(raw_bin, dtype=">f4").reshape(height, width)
            np_data = np_data.byteswap().view(np_data.dtype.newbyteorder())

            # Convert to torch tensor. 
            # For memory safety, you might keep these on CPU if too large for GPU:
            ifg_tensors.append(torch.from_numpy(np_data).to(device))

    # 3) Read the PSC candidates from pscands_ij (HDF5)
    #    We expect a dataset called 'data' with shape (N, 3): (index, y, x)
    with h5py.File(pscands_ij, "r") as ij_hdf:
        ij_data = numpy.array(ij_hdf["data"])  # shape = (N, 3)

    # Separate row/col indices (move them to device if you plan on GPU gather)
    y_coords = torch.from_numpy(ij_data[:, 1]).long().to(device)
    x_coords = torch.from_numpy(ij_data[:, 2]).long().to(device)

    num_points = len(ij_data)
    num_ifgs = len(ifg_tensors)

    # 4) Create the output HDF5 dataset (on CPU) with space for all points/ifgs.
    #    We'll fill it in batches to avoid huge GPU memory usage.
    with h5py.File(pscands_ll, "w") as ll_hdf:
        ll_dataset = ll_hdf.create_dataset(
            "data",
            (num_points, num_ifgs),
            maxshape=(None, num_ifgs),
            dtype="d",
            chunks=True,
        )
        
        # 5) Batch processing over PSC points.
        #    We gather only a slice of PSC row/col coords at a time.
        for start_idx in range(0, num_points, batch_size):
            end_idx = min(start_idx + batch_size, num_points)
            batch_size_actual = end_idx - start_idx

            # Slice the row/col for this batch
            batch_y = y_coords[start_idx:end_idx]
            batch_x = x_coords[start_idx:end_idx]

            # We'll create a temporary array on the GPU or CPU
            # to store the per-IFG results for this batch
            batch_result = torch.empty((batch_size_actual, num_ifgs), dtype=torch.float32, device=device)

            # Gather for each ifg tensor
            for ifg_index, ifg_tensor in enumerate(ifg_tensors):
                batch_result[:, ifg_index] = ifg_tensor[batch_y, batch_x]

            # Move results back to CPU as float64 for writing
            batch_np = batch_result.cpu().numpy().astype(numpy.float64)
            ll_dataset[start_idx:end_idx, :] = batch_np

    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "End"))