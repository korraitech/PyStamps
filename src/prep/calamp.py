import numpy
import torch

def load_file_paths(file_path: str) -> list[str]:
    """Load SLC file paths from input file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def swap_bytes_for_complex(data: torch.Tensor) -> torch.Tensor:
    """Swap bytes for complex float data."""
    data_bytes = data.view(torch.uint8).reshape(-1, 8)
    data_bytes = data_bytes[:, [3, 2, 1, 0, 7, 6, 5, 4]].clone()
    return data_bytes.view(torch.complex64).reshape(data.shape)

def calculate_amplitude_calibration(file_path: str, width: int, device: torch.device) -> str:
    """Calculate amplitude calibration constant for a single SLC file."""
    data = numpy.fromfile(file_path, dtype=numpy.complex64)
    data = torch.from_numpy(data).to(device)  # Move data to the specified device

    # Swap bytes for complex data
    data = swap_bytes_for_complex(data)
    amplitudes = torch.abs(data)
    
    valid_mask = amplitudes > 0.001
    valid_amplitudes = amplitudes[valid_mask]
    
    num_valid_pixels = valid_mask.sum().item()
    
    if num_valid_pixels > 0:
        calibration_factor = valid_amplitudes.sum().item() / num_valid_pixels
    else:
        print(f"WARNING: SLC {file_path} has ZERO mean amplitude")
        calibration_factor = 0.0
        
    return f"{file_path} {calibration_factor}"

def run_calamp(calamp_in: str, width: int, calamp_out: str):
    """Run the calibration process on the input files."""
    print("Running calamp ...\t[{}]".format(calamp_in))
    
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    paths = load_file_paths(calamp_in)
    
    calibration_results = [calculate_amplitude_calibration(path, width, device) for path in paths]
    calibration_results.sort()
    
    with open(calamp_out, 'w') as file:
        for result in calibration_results:
            file.write(f"{result}\n")
