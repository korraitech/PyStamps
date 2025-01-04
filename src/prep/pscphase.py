import h5py
import numpy as np
import struct
def run_pscphase(patch_id:str, pscphase_in:str, pscands_ij:str, pscands_ph:str):
    print(f"Running pscphase ...[{patch_id}]")
    
    ifg_filenames = []
    with open(pscphase_in, 'r') as parmfile:
        width = int(parmfile.readline().strip())
        ifg_filenames = parmfile.read().splitlines()

    with h5py.File(pscands_ij, 'r') as ij_hdf, \
         h5py.File(pscands_ph, 'w') as ph_hdf:
        num_points = len(ij_hdf['data'])
        ph_dataset = ph_hdf.create_dataset('data', (num_points, len(ifg_filenames)), 
                                         maxshape=(None, len(ifg_filenames)), dtype=np.complex64, chunks=True)
        
        for i, ifg_filename in enumerate(ifg_filenames):
            # Create temporary array for current interferogram
            temp_array = np.zeros(num_points, dtype=np.complex64)
            
            with open(ifg_filename, 'rb') as ifgfile:
                header = ifgfile.read(32)
                if struct.unpack('>l', header[:4])[0] != 0x59a66a95:
                    ifgfile.seek(0)
                
                for index, (_, y, x) in enumerate(ij_hdf['data']):
                    xyaddr = (y * width + x) * 8
                    ifgfile.seek(xyaddr)
                    ifg_pixel = ifgfile.read(8)
                    real, imag = struct.unpack('>ff', ifg_pixel)
                    temp_array[index] = complex(real, imag)

            ph_dataset[:,i] = temp_array
