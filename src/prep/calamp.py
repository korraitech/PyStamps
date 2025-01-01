#!/usr/bin/env python3
import sys
import numpy as np

def is_invalid_argument_count(argc: int) -> bool:
    """Check if the number of command line arguments is valid."""
    if argc < 3:
        print("Usage: calamp parmfile.in width parmfile.out")
        print("  parmfile.in(input) SLC file names (complex float)")
        print("  width  width of SLCs")
        print("  parmfile.out(output) SLC file names and calibration constants")
        return True
    return False

def load_file_paths(file_path: str) -> list[str]:
    """Load SLC file paths from input file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def swap_bytes_for_complex(data: np.ndarray) -> np.ndarray:
    """Swap bytes for complex float data."""
    data_bytes = np.ascontiguousarray(data).view(np.uint8).reshape(-1, 8)
    data_bytes = data_bytes[:, [3, 2, 1, 0, 7, 6, 5, 4]].copy()
    return data_bytes.view(np.complex64).reshape(data.shape)

def calculate_amplitude_calibration(file_path: str, width: int) -> str:
    """Calculate amplitude calibration constant for a single SLC file."""
    data = np.fromfile(file_path, dtype=np.complex64)
    data = swap_bytes_for_complex(data)
    amplitudes = np.abs(data)
    
    valid_mask = amplitudes > 0.001
    valid_amplitudes = amplitudes[valid_mask]
    
    num_valid_pixels = valid_mask.sum()
    
    if num_valid_pixels > 0:
        calibration_factor = valid_amplitudes.sum() / num_valid_pixels
    else:
        print(f"WARNING: SLC {file_path} has ZERO mean amplitude")
        calibration_factor = 0.0
        
    return f"{file_path} {calibration_factor}"

def run_calamp(calamp_in: str, width: int, calamp_out: str):
    """Run the calibration process on the input files."""
    print("Running calamp ...\t[{}]".format(calamp_in))
    
    paths = load_file_paths(calamp_in)
    
    calibration_results = [calculate_amplitude_calibration(path, width) for path in paths]
    calibration_results.sort()
    
    with open(calamp_out, 'w') as file:
        for result in calibration_results:
            file.write(f"{result}\n")

def main():
    if is_invalid_argument_count(len(sys.argv)):
        return
        
    width = int(sys.argv[2])
    calamp_out = sys.argv[3] if len(sys.argv) >= 4 else "parmfile.out"
    
    run_calamp(sys.argv[1], width, calamp_out)

if __name__ == "__main__":
    main()
