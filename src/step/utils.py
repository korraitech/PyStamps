import h5py as h5
import numpy as np

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

def load_h5(filename) -> dict:
    """
    Load .h5 file using h5py.
    """ 
    with h5.File(filename, 'r') as f:
        return {k: np.array(v) for k, v in f.items()}