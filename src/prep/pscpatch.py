import numpy
import h5py
import struct
from ..logger import appLogger
from ..misc import get_module_info

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

def process_patch_data_in_batches(
    amp_files: list[tuple[str, float]], 
    coords: tuple[int, int, int, int],
    width: int,
    batch_lines: int = 1024
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Process patch data in batches using numpy.
    Instead of returning per-file amplitudes, we only return:
      1. The sum of amplitudes at each pixel.
      2. The sum of squared amplitudes at each pixel.
      3. The minimum amplitude at each pixel (to replicate the check for any pixel <= threshold).
    """

    rg_start, rg_end, az_start, az_end = coords
    patch_width = rg_end - rg_start + 1
    patch_lines = az_end - az_start + 1
    num_files = len(amp_files)

    # Allocate CPU arrays for accumulations
    sum_amp_cpu = numpy.zeros((patch_lines, patch_width), dtype=numpy.float32)
    sum_amp_sq_cpu = numpy.zeros((patch_lines, patch_width), dtype=numpy.float32)
    min_amp_cpu = numpy.full((patch_lines, patch_width), numpy.inf, dtype=numpy.float32)

    for i, (fname, calib) in enumerate(amp_files):
        # Read one file at a time, in smaller line-batches
        with open(fname, 'rb') as f:
            header = f.read(32)
            # If magic number check fails, reset to start
            if struct.unpack('>l', header[:4])[0] != 0x59a66a95:
                f.seek(0)

            # Move file pointer to patch start
            f.seek((az_start) * width * 8 + (rg_start) * 8)

            lines_remaining = patch_lines
            start_line = 0

            while lines_remaining > 0:
                # Number of lines in this batch
                lines_to_read = min(batch_lines, lines_remaining)

                # Read the raw complex data for this batch
                data_batch = numpy.zeros(
                    (lines_to_read, patch_width), dtype=numpy.complex64
                )
                for line in range(lines_to_read):
                    data_line = numpy.fromfile(f, dtype=numpy.complex64, count=patch_width)
                    data_batch[line] = data_line
                    # Skip remainder in the file
                    f.seek((width - patch_width) * 8, 1)

                # Byte-swap if needed
                data_batch = numpy.array(data_batch.byteswap())

                # Calculate amplitude using numpy
                amplitude = numpy.abs(data_batch) / calib

                # Update accumulations
                sum_amp_cpu[start_line:start_line + lines_to_read] += amplitude
                sum_amp_sq_cpu[start_line:start_line + lines_to_read] += amplitude ** 2
                min_amp_cpu[start_line:start_line + lines_to_read] = numpy.minimum(
                    min_amp_cpu[start_line:start_line + lines_to_read],
                    amplitude
                )

                # Advance
                start_line += lines_to_read
                lines_remaining -= lines_to_read

    # Now compute D_sq = num_files * (sum_amp_sq) / (sum_amp^2) - 1
    # Watch out for zeros in sum_amp
    with numpy.errstate(divide="ignore", invalid="ignore"):
        D_sq_cpu = (num_files * sum_amp_sq_cpu) / (sum_amp_cpu ** 2) - 1
        # Where sum_amp_cpu == 0, set D_sq to a large positive number (or NaN -> just handle it)
        D_sq_cpu[numpy.isnan(D_sq_cpu) | numpy.isinf(D_sq_cpu)] = 9999999

    return min_amp_cpu, sum_amp_cpu, D_sq_cpu


def find_ps_candidates_batched(
    min_amp: numpy.ndarray,
    sum_amp: numpy.ndarray, 
    D_sq: numpy.ndarray, 
    D_thresh: float, 
    coords: tuple[int, int, int, int], 
    pscands_ij: str,
    pscands_da: str,
    pscands_ma: str
) -> None:
    """
    Identify PS candidates and write results to HDF5, using:
      -- min_amp: the minimum amplitude across all images (to replicate "if any amplitude <= threshold")
      -- sum_amp: sum of amplitudes across all images
      -- D_sq: computed phase-stability statistic
    """

    rg_start, az_start = coords[0], coords[2]
    patch_lines, patch_width = sum_amp.shape

    # Same mask logic
    if D_thresh >= 0:
        ps_mask = (D_sq < D_thresh**2) & (sum_amp > 0)
    else:
        ps_mask = (D_sq >= D_thresh**2) & (sum_amp > 0)

    pscid = 0
    ij_data = []
    daout_data = []

    # Create separate HDF5 files for each dataset
    with h5py.File(pscands_ij, 'w') as ij_hdf, \
         h5py.File(pscands_da, 'w') as da_hdf, \
         h5py.File(pscands_ma, 'w') as ma_hdf:

        ps_mask_flat = ps_mask.flatten()
        D_sq_flat = D_sq.flatten()

        az_indices = numpy.arange(az_start, az_start + patch_lines)
        rg_indices = numpy.arange(rg_start, rg_start + patch_width)
        az_grid, rg_grid = numpy.meshgrid(az_indices, rg_indices, indexing='ij')
        az_flat = az_grid.flatten()
        rg_flat = rg_grid.flatten()

        # Flatten min_amp to replicate "if any amplitude <= 0.00005"
        min_amp_flat = min_amp.flatten()

        valid_indices = numpy.where(ps_mask_flat)[0]

        for idx in valid_indices:
            # check if min amplitude was below threshold => skip
            if min_amp_flat[idx] <= 0.00005:
                continue
            az_val = az_flat[idx]
            rg_val = rg_flat[idx]
            ij_data.append([pscid, az_val, rg_val])
            daout_data.append(numpy.sqrt(D_sq_flat[idx]))
            pscid += 1
            
        # Write out results
        ma_dataset = ma_hdf.create_dataset(
            'data', (patch_lines, patch_width),
            dtype='f', chunks=True
        )
        ma_dataset[:, :] = sum_amp

        ij_dataset = ij_hdf.create_dataset(
            'data', (len(ij_data), 3),
            maxshape=(None, 3),
            dtype='i', chunks=True
        )
        ij_dataset[:] = ij_data

        da_dataset = da_hdf.create_dataset(
            'data', (len(daout_data),),
            maxshape=(None,),
            dtype='f', chunks=True
        )
        da_dataset[:] = daout_data


def run_pscpatch(
    patch_id: str, 
    selpsc_in: str, 
    patch_in: str,
    pscands_ij: str,
    pscands_da: str,
    pscands_ma: str
) -> None:
    """
    Run the PSC patch process using only numpy.
    """
    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "Start"))

    D_thresh, width, amp_files = read_parmfile(selpsc_in)
    coords = read_patch_coords(patch_in)

    # Compute data in a streaming/batch fashion using numpy
    min_amp, sum_amp, D_sq = process_patch_data_in_batches(
        amp_files, coords, width
    )

    # Find PS candidates using the aggregated results
    find_ps_candidates_batched(
        min_amp, sum_amp, D_sq, D_thresh, coords,
        pscands_ij, pscands_da, pscands_ma
    )
    appLogger.info(">>>>>>>>>>>>>>>> {} || {} {}".format(get_module_info(),patch_id, "End"))