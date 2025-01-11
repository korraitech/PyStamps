import h5py
import numpy
import torch
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
        ij_array = numpy.array(ij_hdf['data'])  # shape: (N, 3) => [id, y, x]
    y_coords = ij_array[:, 1].astype(numpy.int64)
    x_coords = ij_array[:, 2].astype(numpy.int64)
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

        # Device for PyTorch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # We need to map each row index into chunks:
        # For each chunk, we read chunk_height rows, figure out which PSCs are in that row range,
        # flatten them, gather from that chunk, and write to keep the original ordering.

        # We'll keep an array of final PSC height in CPU memory to write chunk by chunk.
        # If memory is still too big, you could store partial results to disk, but
        # typically the PSC array is much smaller than the entire image.
        # For truly huge PSC arrays, you'd do a similar chunk approach for writing.

        # This approach uses indexing back into ht_dset by PSC index in the original ordering
        # so that we preserve the final shape and order in the "data" dataset.
        
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

            # Convert buffer to float32
            chunk_numpy = numpy.frombuffer(chunk_buffer, dtype='>f4').astype('float32')
            # Move to GPU if available
            chunk_torch = torch.from_numpy(chunk_numpy).to(device)

            # Indices of PSCs in this chunk
            # (y in [current_row_start, current_row_end))
            in_chunk_mask = (y_coords >= current_row_start) & (y_coords < current_row_end)
            chunk_indices = numpy.where(in_chunk_mask)[0]  # PSC subset indices

            if len(chunk_indices) > 0:
                # local row = y - current_row_start
                local_y = y_coords[chunk_indices] - current_row_start
                local_x = x_coords[chunk_indices]
                local_linidx = (local_y * width) + local_x
                # Move to GPU, gather
                local_linidx_torch = torch.from_numpy(local_linidx).to(device)
                result_torch = chunk_torch[local_linidx_torch]
                result_cpu = result_torch.cpu().numpy()
                # Write back to HDF5 dataset at the correct PSC candidate positions
                ht_dset[chunk_indices] = result_cpu

            current_row_start = current_row_end

    ifgfile.close()
    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "End"))
