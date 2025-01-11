import numpy
import torch
from ..logger import appLogger
from ..misc import get_module_info

def load_file_paths(file_path: str) -> list[str]:
    """
    Load SLC file paths from input file.
    """
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def swap_bytes_for_complex(data: torch.Tensor) -> torch.Tensor:
    """
    Swap bytes for complex float data.
    """
    data_bytes = data.view(torch.uint8).reshape(-1, 8)
    data_bytes = data_bytes[:, [3, 2, 1, 0, 7, 6, 5, 4]].clone()
    return data_bytes.view(torch.complex64).reshape(data.shape)

def calculate_amplitude_calibration(file_path: str, width: int, device: torch.device) -> str:
    """
    Calculate amplitude calibration constant for a single SLC file.
    Uses the same logic as the batch version, but processes only one file.
    """
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

def calculate_amplitude_calibration_batch(file_paths: list[str], width: int, device: torch.device) -> list[str]:
    """
    Calculate amplitude calibration constants for multiple SLC files in parallel.
    This version uses batched tensors and vectorized operations in PyTorch
    to optimize performance, while preserving the original logic.
    """
    results = []
    
    # Process files in batches to manage memory usage on large datasets.
    # Increase this value if you have enough GPU memory to handle larger batches.
    batch_size = 64
    
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i : i + batch_size]

        # Load all files in the current batch
        batch_data = []
        for path in batch_paths:
            raw_data = numpy.fromfile(path, dtype=numpy.complex64)
            batch_data.append(torch.from_numpy(raw_data).to(device))

        # Stack all data Tensors in this batch (shape: [batch_size, num_samples])
        batch_tensor = torch.stack(batch_data)

        # Swap bytes for the entire batch at once
        batch_tensor = swap_bytes_for_complex(batch_tensor)

        # Compute amplitudes
        batch_amplitudes = torch.abs(batch_tensor)

        # Create a mask of valid amplitudes
        valid_mask = batch_amplitudes > 0.001

        # Vectorized sum of valid amplitudes per file in the batch
        sum_amplitudes = torch.where(valid_mask, batch_amplitudes, torch.tensor(0.0, device=device)).sum(dim=1)

        # Count valid pixels per file in the batch
        valid_counts = valid_mask.sum(dim=1)

        # Finish up by assigning a calibration factor for each file
        for idx, path in enumerate(batch_paths):
            if valid_counts[idx].item() > 0:
                calibration_factor = sum_amplitudes[idx].item() / valid_counts[idx].item()
            else:
                print(f"WARNING: SLC {path} has ZERO mean amplitude")
                calibration_factor = 0.0
            results.append(f"{path} {calibration_factor}")

    return results

def run_calamp(calamp_in: str, width: int, calamp_out: str):
    """
    Run the calibration process on the input files using optimized PyTorch operations.
    """
    appLogger.info(
        ">>>>>>>>>>>>>>>> {}\t\t|| {} {} {}".format(
            get_module_info(), calamp_in, width, calamp_out
        )
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = load_file_paths(calamp_in)
    
    # Use optimized batch processing
    calibration_results = calculate_amplitude_calibration_batch(paths, width, device)
    calibration_results.sort()
    
    # Write out results
    with open(calamp_out, 'w') as file:
        for result in calibration_results:
            file.write(f"{result}\n")
