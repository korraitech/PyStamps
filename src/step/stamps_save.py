import os
import h5py

def append_to_hdf5(save_path, save_dict):
    print('Appending to existing HDF5 file...')
    with h5py.File(save_path, 'a') as f:
        for key, value in save_dict.items():
            if key in f:
                del f[key]
            f.create_dataset(key, data=value)

def save_to_hdf5(save_path, save_dict):
    print('Saving in HDF5 format')
    with h5py.File(save_path, 'w') as f:
        for key, value in save_dict.items():
            #print(f"key,value,type {key} {value} {type(value)}")
            f.create_dataset(key, data=value)

def stamps_save(output_dir:str,file_name:str, **kwargs):
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
