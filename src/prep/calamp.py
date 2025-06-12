#########################################################################
#   Copyright 2025 - 2025, KorrAI                                       #
#   ALL RIGHTS RESERVED.                                                #
#   This file is subject to the full copyright and disclaimer notice    #
#   included in a separate file in this directory.                      #
#########################################################################
#                                                                       #
#   This file contains the implementation of calamp.                    #
#                                                                       #
#########################################################################


import numpy as np
from ..logger import appLogger
from ..misc import get_module_info

def load_file_paths(file_path: str) -> list[str]:
    """
    Load SLC file paths from input file.
    """
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def calculate_amplitude_calibration_chunked(file_path: str, chunk_size: int = 1_000_000) -> float:
    """
    Compute the amplitude calibration factor for a single SLC file by reading it in chunks.
    This prevents loading the entire file into memory at once.
    """
    # Partial sums/counters
    sum_amplitudes = 0.0
    valid_pixels = 0

    # Open file using memmap or direct open in binary
    with open(file_path, 'rb') as f:
        # Read until no more data
        while True:
            # Read chunk_size of complex64 samples (each sample is 8 bytes)
            chunk = np.fromfile(f, dtype=np.complex64, count=chunk_size)
            if chunk.size == 0:
                break

            # Swap (reverse) bytes if needed
            data = np.array(chunk.byteswap())

            # Compute amplitudes
            amplitudes = np.abs(data)

            # Filter
            valid_mask = amplitudes > 0.001
            sum_amplitudes += np.sum(amplitudes[valid_mask])
            valid_pixels += np.sum(valid_mask)

    if valid_pixels > 0:
        calibration_factor = sum_amplitudes / valid_pixels
    else:
        print(f"WARNING: SLC {file_path} has ZERO mean amplitude")
        calibration_factor = 0.0

    return calibration_factor

def calculate_amplitude_calibration_batch(file_paths: list[str], width: int, chunk_size: int = 1_000_000) -> list[str]:
    """
    Calculate amplitude calibration constants by processing each file chunk-by-chunk.
    This method avoids memory issues by never loading the entire dataset for all files at once.
    """
    results = []
    for path in file_paths:
        calibration_factor = calculate_amplitude_calibration_chunked(path, chunk_size)
        results.append(f"{path} {calibration_factor}")
    return results

def run_calamp(calamp_in: str, width: int, calamp_out: str):
    """
    Run the calibration process on the input files using chunk-based loading to avoid memory issues.
    """
    appLogger.info(">>>>>>>>>>>>>>>> {} || {}".format(get_module_info(),"Start"))
    
    # Load file paths
    paths = load_file_paths(calamp_in)

    # Perform calibration chunk-by-chunk
    # Adjust 'chunk_size' to tune memory/performance:
    calibration_results = calculate_amplitude_calibration_batch(paths, width, chunk_size=1_000_000)

    # Sort for consistent output
    calibration_results.sort()
    
    # Write out results
    with open(calamp_out, 'w') as file:
        for result in calibration_results:
            file.write(f"{result}\n")
    appLogger.info(">>>>>>>>>>>>>>>> {} || {}".format(get_module_info(),"End"))