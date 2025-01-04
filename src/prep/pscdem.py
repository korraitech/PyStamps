import h5py
import numpy as np

def run_pscdem(patch_id:str,pscphase_in:str, pscands_ij:str, pscands_ht:str):
    print(f"Running pscdem ...[{patch_id}]")

    with open(pscphase_in, 'r') as parmfile:
        width = int(parmfile.readline().strip())
        ifgfilename = parmfile.readline().strip()

    ifgfile = open(ifgfilename, 'rb')
    header = ifgfile.read(32)
    if int.from_bytes(header[:4], 'little') == 0x59a66a95:
        print("pscdem: sun raster file - skipping header")
    else:
        ifgfile.seek(0)
    
    with h5py.File(pscands_ij, 'r') as ij_hdf, \
         h5py.File(pscands_ht, 'w') as ht_hdf:
        ht_data = []
        for _, y, x in ij_hdf['data']:
            xyaddr = (y * width + x) * 4 # 4 bytes for float
            ifgfile.seek(xyaddr)
            ht_data.append(float(np.frombuffer(ifgfile.read(np.dtype('>f4').itemsize), dtype='>f4')))
        ht_dataset = ht_hdf.create_dataset('data', (len(ht_data),), 
                                           maxshape=(None, ), dtype='f', chunks=True)
        ht_dataset[:] = ht_data
    
    ifgfile.close()