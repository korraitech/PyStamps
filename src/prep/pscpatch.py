import numpy as np
from typing import List, Tuple
import struct
import h5py

def byteswap_complex(data: np.ndarray) -> np.ndarray:
    """Swap bytes for complex float data."""
    return np.array(data.byteswap())

def read_parmfile(filename: str) -> Tuple[float, int, List[Tuple[str, float]]]:
    """Read parameter file and return threshold, width and calibration factors."""
    with open(filename) as file:
        D_thresh = float(file.readline())
        width = int(file.readline())
        amp_files = [
            (fname, float(calib))
            for line in file if line.strip()
            for fname, calib in [line.strip().split()]
        ]
    return D_thresh, width, amp_files

def read_patch_coords(filename: str) -> Tuple[int, int, int, int]:
    """Read patch coordinates."""
    with open(filename) as file:
        return tuple(int(file.readline()) for _ in range(4))

def process_patch_data(amp_files: List[Tuple[str, float]], 
                       coords: Tuple[int, int, int, int],
                       width: int,
                       D_thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process patch data and calculate amplitudes and statistics."""
    rg_start, rg_end, az_start, az_end = coords
    patch_width = rg_end - rg_start + 1
    patch_lines = az_end - az_start + 1
    num_files = len(amp_files)
    
    patch_data = np.zeros((num_files, patch_lines, patch_width), dtype=np.complex64)
    calib_factors = np.array([f[1] for f in amp_files])
    
    for i, (fname, _) in enumerate(amp_files):
        with open(fname, 'rb') as file:
            header = file.read(32)
            if struct.unpack('>l', header[:4])[0] != 0x59a66a95:
                file.seek(0)
            file.seek((az_start - 1) * width * 8 + (rg_start - 1) * 8)
            for line in range(patch_lines):
                data = np.fromfile(file, dtype=np.complex64, count=patch_width)
                patch_data[i, line] = data
                file.seek((width - patch_width) * 8, 1)
    
    patch_data = byteswap_complex(patch_data)
    amplitudes = np.abs(patch_data)
    sum_amp = np.sum(amplitudes / calib_factors[:, None, None], axis=0)
    sum_amp_sq = np.sum((amplitudes / calib_factors[:, None, None])**2, axis=0)
    D_sq = num_files * sum_amp_sq / (sum_amp**2) - 1
    
    return amplitudes, sum_amp, D_sq

def find_ps_candidates(amplitudes: np.ndarray, sum_amp: np.ndarray, D_sq: np.ndarray, 
                       D_thresh: float, coords: Tuple[int, int, int, int], 
                       pscands_ij:str,
                       pscands_da:str,
                       pscands_ma:str) -> None:
    """Identify PS candidates and write results to HDF5."""
    rg_start, az_start = coords[0], coords[2]
    patch_lines, patch_width = sum_amp.shape
    ps_mask = (D_sq < D_thresh**2) & (sum_amp > 0) if D_thresh >= 0 else (D_sq >= D_thresh**2) & (sum_amp > 0)
    
    pscid = 0
    ij_data = []
    jiname_data = []
    ijname0_data = []
    daout_data = []

    # Create separate HDF5 files for each dataset
    with h5py.File(pscands_ij, 'w') as ij_hdf, \
         h5py.File(pscands_da, 'w') as da_hdf, \
         h5py.File(pscands_ma, 'w') as ma_hdf:

        # Create datasets in their respective files
        #ij_dataset = ij_hdf.create_dataset('data', (0, 3), maxshape=(None, 3), dtype='i', chunks=True)
        #da_dataset = da_hdf.create_dataset('data', (0,), maxshape=(None,), dtype='d', chunks=True)
        #ma_dataset = ma_hdf.create_dataset('data', (patch_lines, patch_width), dtype='d')

        # jiname_dataset = jiname_hdf.create_dataset('data', (0, 2), maxshape=(None, 2), dtype='i', chunks=True)
        # ijname0_dataset = ijname0_hdf.create_dataset('data', (0, 3), maxshape=(None, 3), dtype='i', chunks=True)

        ps_mask_flat = ps_mask.flatten()
        amplitudes_flat = amplitudes.reshape(amplitudes.shape[0], -1)
        D_sq_flat = D_sq.flatten()

        # Precompute az and rg indices
        az_indices = np.arange(az_start - 1, az_start - 1 + patch_lines)
        rg_indices = np.arange(rg_start - 1, rg_start - 1 + patch_width)
        az_grid, rg_grid = np.meshgrid(az_indices, rg_indices, indexing='ij')
        az_flat = az_grid.flatten()
        rg_flat = rg_grid.flatten()

        # Filter indices where ps_mask is True
        valid_indices = np.where(ps_mask_flat)[0]

        # Process valid indices
        for idx in valid_indices:
            pscid += 1
            az = az_flat[idx]
            rg = rg_flat[idx]
            if np.any(amplitudes_flat[:, idx] <= 0.00005):
                ijname0_data.append([pscid, az, rg])
            else:
                ij_data.append([pscid, az, rg])
                jiname_data.append([rg, az])
                daout_data.append(np.sqrt(D_sq_flat[idx]))

        # Update ma_dataset
        ma_dataset = ma_hdf.create_dataset('data', (patch_lines, patch_width), dtype='d',chunks=True)
        ma_dataset[:, :] = sum_amp

        # Write collected data to datasets
        ij_dataset = ij_hdf.create_dataset('data', (len(ij_data), 3), maxshape=(None, 3), dtype='i', chunks=True)
        ij_dataset[:] = ij_data
        # if jiname_data:
        #     jiname_dataset.resize((len(jiname_data), 2))
        #     jiname_dataset[:] = jiname_data
        # if ijname0_data:
        #     ijname0_dataset.resize((len(ijname0_data), 3))
        #     ijname0_dataset[:] = ijname0_data
        # if daout_data:
        da_dataset = da_hdf.create_dataset('data', (len(daout_data),), maxshape=(None,), dtype='d', chunks=True)
        da_dataset[:] = daout_data

def run_pscpatch(patch_id: str, selpsc_in: str, patch_in: str,
        pscands_ij:str,
        pscands_da:str,
        pscands_ma:str
        ) -> None:
    """Run the PSC patch process."""
    print(f"Running pscpatch ...[{patch_id}]")
    
    D_thresh, width, amp_files = read_parmfile(selpsc_in)
    coords = read_patch_coords(patch_in)
    amplitudes, sum_amp, D_sq = process_patch_data(amp_files, coords, width, D_thresh)
    find_ps_candidates(amplitudes, sum_amp, D_sq, D_thresh, coords, pscands_ij,
        pscands_da,
        pscands_ma)
