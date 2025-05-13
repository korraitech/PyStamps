import h5py as hp
import numpy as np
from scipy.signal.windows import gaussian
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

def gaussian1D(w:int, alpha:float = 2.5):
    std_dev = (w - 1) / (2 * alpha)
    return gaussian(w, std=std_dev, sym=True)

def gaussian2D(w:int, alpha:float = 2.5):
    gaussian_1d = gaussian1D(w,alpha)
    return np.outer(gaussian_1d, gaussian_1d)
