import h5py
import numpy as np
import struct

def run_psclonlat(patch_id:str, psclatlon_in:str, 
                  pscands_ij:str, pscands_ll:str):
    print(f"Running psclonlat ...[{patch_id}]")

    lonlat_files = []
    with open(psclatlon_in, 'r') as parmfile:
        width = int(parmfile.readline().strip())
        lonlat_files.append(parmfile.readline().strip())
        lonlat_files.append(parmfile.readline().strip())

    ifgfiles = []
    for lonlat_file in lonlat_files:
        file =  open(lonlat_file, 'rb')
        header = file.read(32)
        if struct.unpack('>l', header[:4])[0] != 0x59a66a95:
            file.seek(0)
        ifgfiles.append(file)

    with h5py.File(pscands_ij, 'r') as ij_hdf, \
         h5py.File(pscands_ll, 'w') as ll_hdf:
        ll_dataset = ll_hdf.create_dataset('data', (len(ij_hdf['data']), 
            len(ifgfiles)), maxshape=(None, len(ifgfiles)), dtype='d', chunks=True)
        index = 0
        for _, y, x in ij_hdf['data']:
            xyaddr = (y * width + x) * 4    # 4 bytes for float
            for ifg_index, ifgfile in enumerate(ifgfiles):
                ifgfile.seek(xyaddr)
                ifg_pixel = ifgfile.read(4)
                ll_dataset[index, ifg_index] = np.frombuffer(ifg_pixel, dtype='>f4')
            index += 1
    
    for lonlat_file in ifgfiles:
        lonlat_file.close()