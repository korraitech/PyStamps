import numpy
import h5py
import torch
import struct
from ..logger import appLogger
from ..misc import get_module_info

def byteswap_complex(data: numpy.ndarray) -> numpy.ndarray:
    """
    Swap bytes for complex float data (Numpy-based).
    """
    return numpy.array(data.byteswap())

def read_parmfile(filename: str) -> tuple[float, int, list[tuple[str, float]]]:
    """
    Read parameter file and return threshold, width and calibration factors.
    """
    with open(filename) as file:
        D_thresh = float(file.readline())
        width = int(file.readline())
        amp_files = [
            (fname, float(calib))
            for line in file if line.strip()
            for fname, calib in [line.strip().split()]
        ]
    return D_thresh, width, amp_files

def read_patch_coords(filename: str) -> tuple[int, int, int, int]:
    """
    Read patch coordinates.
    """
    with open(filename) as file:
        return tuple(int(file.readline()) for _ in range(4))

def process_patch_data(
    amp_files: list[tuple[str, float]], 
    coords: tuple[int, int, int, int],
    width: int,
    device: torch.device
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Process patch data, converting computations to PyTorch for possible GPU acceleration.
    """
    rg_start, rg_end, az_start, az_end = coords
    patch_width = rg_end - rg_start + 1
    patch_lines = az_end - az_start + 1
    num_files = len(amp_files)

    # Allocate NumPy array first (to read from file), then will convert to PyTorch
    patch_data_np = numpy.zeros((num_files, patch_lines, patch_width), dtype=numpy.complex64)
    calib_factors = numpy.array([f[1] for f in amp_files])

    # Read the data from file in the same way
    for i, (fname, _) in enumerate(amp_files):
        with open(fname, 'rb') as file:
            header = file.read(32)
            if struct.unpack('>l', header[:4])[0] != 0x59a66a95:
                file.seek(0)
            # Seek to the start of the patch
            file.seek((az_start - 1) * width * 8 + (rg_start - 1) * 8)
            for line in range(patch_lines):
                data_line = numpy.fromfile(file, dtype=numpy.complex64, count=patch_width)
                patch_data_np[i, line] = data_line
                # Move the file pointer to skip the remainder
                file.seek((width - patch_width) * 8, 1)

    # Byte-swap in NumPy if needed
    patch_data_np = byteswap_complex(patch_data_np)

    # Move patch_data and calib_factors to torch
    patch_data = torch.from_numpy(patch_data_np).to(device)
    calib_factors_torch = torch.from_numpy(calib_factors).float().to(device)

    # Compute amplitudes = absolute value of complex data
    # (PyTorch complex is still evolving; treat real/imag as float pairs if needed.)
    # In many builds, abs() will handle complex64 natively, but if not, do manual sqrt(real^2 + imag^2).
    amplitudes = torch.abs(patch_data)

    # Dividing by calibration factors
    # calib_factors_torch has shape [num_files], so we reshape to broadcast
    den = calib_factors_torch.view(num_files, 1, 1)
    amp_div = amplitudes / den

    # sum_amp = numpy.sum(amplitudes / calib_factors[:, None, None], axis=0)
    sum_amp_torch = amp_div.sum(dim=0)
    # sum_amp_sq = numpy.sum((amplitudes / calib_factors[:, None, None])**2, axis=0)
    sum_amp_sq_torch = (amp_div ** 2).sum(dim=0)

    # D_sq = num_files * sum_amp_sq / (sum_amp**2) - 1
    # Make sure we do it safely where sum_amp != 0
    # (If sum_amp = 0 somewhere, the same would happen in numpy as a divide-by-zero.)
    D_sq_torch = num_files * sum_amp_sq_torch / (sum_amp_torch ** 2) - 1

    # Move results back to CPU NumPy
    amplitudes_cpu = amplitudes.cpu().numpy()
    sum_amp_cpu = sum_amp_torch.cpu().numpy()
    D_sq_cpu = D_sq_torch.cpu().numpy()
    return amplitudes_cpu, sum_amp_cpu, D_sq_cpu

def find_ps_candidates(
    amplitudes: numpy.ndarray, 
    sum_amp: numpy.ndarray, 
    D_sq: numpy.ndarray, 
    D_thresh: float, 
    coords: tuple[int, int, int, int], 
    pscands_ij: str,
    pscands_da: str,
    pscands_ma: str
) -> None:
    """Identify PS candidates and write results to HDF5."""

    rg_start, az_start = coords[0], coords[2]
    patch_lines, patch_width = sum_amp.shape

    # Replicate the same mask logic with NumPy
    if D_thresh >= 0:
        ps_mask = (D_sq < D_thresh**2) & (sum_amp > 0)
    else:
        ps_mask = (D_sq >= D_thresh**2) & (sum_amp > 0)

    pscid = 0
    ij_data = []
    # jiname_data = []
    # ijname0_data = []
    daout_data = []

    # Create separate HDF5 files for each dataset
    with h5py.File(pscands_ij, 'w') as ij_hdf, \
         h5py.File(pscands_da, 'w') as da_hdf, \
         h5py.File(pscands_ma, 'w') as ma_hdf:

        ps_mask_flat = ps_mask.flatten()
        amplitudes_flat = amplitudes.reshape(amplitudes.shape[0], -1)
        D_sq_flat = D_sq.flatten()

        # Precompute az and rg indices
        az_indices = numpy.arange(az_start - 1, az_start - 1 + patch_lines)
        rg_indices = numpy.arange(rg_start - 1, rg_start - 1 + patch_width)
        az_grid, rg_grid = numpy.meshgrid(az_indices, rg_indices, indexing='ij')
        az_flat = az_grid.flatten()
        rg_flat = rg_grid.flatten()

        # Filter indices where ps_mask is True
        valid_indices = numpy.where(ps_mask_flat)[0]

        # Process valid indices
        for idx in valid_indices:
            pscid += 1
            az_val = az_flat[idx]
            rg_val = rg_flat[idx]
            if numpy.any(amplitudes_flat[:, idx] <= 0.00005):
                #ijname0_data.append([pscid, az_val, rg_val])
                pass
            else:
                ij_data.append([pscid, az_val, rg_val])
                #jiname_data.append([rg_val, az_val])
                daout_data.append(numpy.sqrt(D_sq_flat[idx]))

        # Write collected data to datasets
        ma_dataset = ma_hdf.create_dataset('data', (patch_lines, patch_width),
                                           dtype='d', chunks=True)
        ma_dataset[:, :] = sum_amp

        ij_dataset = ij_hdf.create_dataset('data', (len(ij_data), 3),
                                           maxshape=(None, 3),
                                           dtype='i', chunks=True)
        ij_dataset[:] = ij_data

        da_dataset = da_hdf.create_dataset('data', (len(daout_data),),
                                           maxshape=(None,),
                                           dtype='d', chunks=True)
        da_dataset[:] = daout_data

def run_pscpatch(
    patch_id: str, 
    selpsc_in: str, 
    patch_in: str,
    pscands_ij: str,
    pscands_da: str,
    pscands_ma: str
) -> None:
    """Run the PSC patch process with PyTorch optimization."""
    appLogger.info(">>>>>>>>>>>>>>>> {}\t\t|| {} {} {} {} {} {}".format(
        get_module_info(),
        patch_id, selpsc_in, patch_in, pscands_ij,
        pscands_da, pscands_ma
    ))

    # Detect GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D_thresh, width, amp_files = read_parmfile(selpsc_in)
    coords = read_patch_coords(patch_in)

    # Compute data with PyTorch
    amplitudes, sum_amp, D_sq = process_patch_data(
        amp_files, coords, width, device
    )

    # Find PS candidates with the results
    find_ps_candidates(
        amplitudes, sum_amp, D_sq, D_thresh, coords,
        pscands_ij, pscands_da, pscands_ma
    )
