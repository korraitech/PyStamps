import h5py as hp
import numpy as np
import os

def read_lines(path:str) -> list[str]:
    """
    Read a file and return a list of lines.
    """
    lines = []
    with open(path, 'r') as f:
        lines = [line.strip() for line in f]
    return lines

def get_par(path:str) -> dict:
    """
    Read a parameter file and return a dictionary with the parameters.
    """
    field_data = {}
    with open(path) as fp:
        for line in fp.readlines():
            line_data = line.strip().split(':')
            field_data[line_data[0]] = line_data[1].split()
    return field_data

def read_h5(filename:str) -> dict:
    """
    Load .h5 file using h5py.
    """ 
    with hp.File(filename, 'r') as f:
        return {k: np.array(v) for k, v in f.items()}

def append_to_hdf5(save_path:str, save_dict:dict):
    print('Appending to existing HDF5 file...')
    with hp.File(save_path, 'a') as f:
        for key, value in save_dict.items():
            if key in f:
                del f[key]
            f.create_dataset(key, data=value)

def save_to_hdf5(save_path, save_dict):
    print('Saving in HDF5 format')
    with hp.File(save_path, 'w') as f:
        for key, value in save_dict.items():
            f.create_dataset(key, data=value)

def save_h5(output_dir:str,file_name:str, **kwargs):
    """
    Save variables to a file, handling large datasets appropriately.
    
    Parameters:
        output_dir (str, optional): Directory to save output files.
        save_name (str): Name of the file to save to
        **kwargs: Named variables to save (optional)
    """
    save_path = os.path.join(output_dir, file_name)
    save_dict = kwargs

    if os.path.exists(save_path):
        append_to_hdf5(save_path, save_dict)
    else:
        save_to_hdf5(save_path, save_dict)

def writecpx(workdir:str,fname:str, vname:np.array, precision='float', endian='n'):
    """
    Translate the MATLAB writecpx.m into Python.
    Writes the real and imaginary parts of vname to a binary file (fname),
    interleaving them (real, imag, real, imag, ...).
    
    :param fname: Output file name (string)
    :param vname: Complex NumPy array to write
    :param precision: 'float' or 'double' (default 'float'), matches MATLAB usage
    :param endian: Endianness: 'n' for native, 'l' for little, 'b' for big (default 'n')
    """
    # Map MATLAB precisions to NumPy data types
    type_map = {
        'float': 'f4',   # 'float32' in NumPy
        'double': 'f8'   # 'float64' in NumPy
    }

    # Map MATLAB endianness notation to NumPy
    endian_map = {
        'n': '=',  # native
        'l': '<',  # little
        'b': '>'   # big
    }

    # Ensure valid inputs
    if precision not in type_map:
        raise ValueError(f"Unsupported precision '{precision}'. Use 'float' or 'double'.")
    if endian not in endian_map:
        raise ValueError(f"Unsupported endian '{endian}'. Use 'n', 'l', or 'b'.")

    # Construct the NumPy dtype with the requested precision and endianness
    dtype_str = endian_map[endian] + type_map[precision]
    out_dtype = np.dtype(dtype_str)
    
    # Create the interleaved real/imag data
    # MATLAB shape: (rows, 2 * cols), then transposed
    rows, cols = vname.shape
    vname_flt = np.zeros((rows, 2 * cols), dtype=np.float64)
    vname_flt[:, 0::2] = vname.real
    vname_flt[:, 1::2] = vname.imag
    
    # Write to file in binary: replicate fwrite(fid, vname_flt.', 'precision')
    # i.e. we transpose it. Then ensure the correct dtype, then tofile.
    with open(fname, 'wb') as f:
        vname_flt.T.astype(out_dtype).tofile(f)

