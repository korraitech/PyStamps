import h5py
import numpy
import torch
from ..logger import appLogger
from ..misc import get_module_info

def run_pscdem(patch_id: str, pscphase_in: str, pscands_ij: str, pscands_ht: str):
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {} {} {} {}".format(
        get_module_info(), patch_id, pscphase_in, pscands_ij, pscands_ht))

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
    
    # Read all remaining bytes from the file into memory
    file_buffer = ifgfile.read()
    ifgfile.close()

    # Convert the buffer into a big-endian float32 NumPy array, then convert it to native byte order.
    # This avoids PyTorch's error about mismatched byte orders.
    full_data = numpy.frombuffer(file_buffer, dtype='>f4').astype('float32')

    # Move data to a CUDA tensor if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    full_data_tensor = torch.from_numpy(full_data).to(device)

    # Open the PSC candidate indices and create the output HDF5 for heights
    with h5py.File(pscands_ij, 'r') as ij_hdf, h5py.File(pscands_ht, 'w') as ht_hdf:
        # Read the (id, y, x) data into a NumPy array
        ij_array = numpy.array(ij_hdf['data'])
        # Extract the y, x coordinates (ignore the first column if it's just an ID)
        y_coords = ij_array[:, 1].astype(numpy.int64)
        x_coords = ij_array[:, 2].astype(numpy.int64)

        # Compute linear indices for the flattened ifg data array
        # Each pixel is 4 bytes, but we've already converted them into floats
        linear_indices = (y_coords * width) + x_coords

        # Move indices to the same device
        linear_indices_torch = torch.from_numpy(linear_indices).to(device)

        # Use fast GPU-based indexing (all in one shot)
        ht_data_torch = full_data_tensor[linear_indices_torch]

        # Bring results back to CPU to save into the new HDF5 file
        ht_data_cpu = ht_data_torch.cpu().numpy()

        # Create and write to "data" dataset
        ht_hdf.create_dataset(  'data',
                                data=ht_data_cpu,
                                maxshape=(None,),
                                dtype='f',
                                chunks=True)
