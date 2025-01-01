import numpy as np
from scipy import signal
from typing import Optional

def clap_filt(ph: np.ndarray, 
              alpha: float = 0.5, 
              beta: float = 0.1, 
              n_win: int = 32, 
              n_pad: int = 0, 
              low_pass: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Combined Low-pass Adaptive Phase filtering.
    
    Args:
        ph: Input phase array
        alpha: Alpha parameter (default: 0.5)
        beta: Beta parameter (default: 0.1)
        n_win: Window size (default: 32)
        n_pad: Padding size (default: 0)
        low_pass: Low-pass filter (fftshifted) (default: None)
    
    Returns:
        Filtered phase array
    """
    if low_pass is None:
        low_pass = np.zeros((n_win + n_pad, n_win + n_pad))

    ph_out = np.zeros_like(ph)
    n_i, n_j = ph.shape

    n_inc = n_win // 4
    n_win_i = int(np.ceil(n_i / n_inc) - 3)
    n_win_j = int(np.ceil(n_j / n_inc) - 3)

    # Create window function
    x = np.arange(n_win//2)
    X, Y = np.meshgrid(x, x)
    X = X + Y
    wind_func = np.block([[X, np.fliplr(X)], 
                         [np.flipud(X), np.flipud(np.fliplr(X))]])
    wind_func = wind_func + 1e-6  # prevent zero in corners

# correct till here

    # Handle NaN values
    ph = np.nan_to_num(ph, 0)

    # Derive the equivalent standard deviation from MATLAB’s definition.
    # In MATLAB:  w(n) = exp( -1/2 * [ alpha * ((n - (M-1)/2) / (M/2)) ]^2 )
    # In SciPy:   w(n) = exp( -(n - center)^2 / (2 * std^2) )
    # Matching exponents => std = (M/2) / alpha
    gaussian_window_size = 7
    # alpha = 2.5
    gaussian_alpha = 2.92
    gaussian_std = (gaussian_window_size / 2.0) / gaussian_alpha   # = 0.4 for M=2, alpha=2.5
    # Create Gaussian window
    gaussian_window = signal.windows.gaussian(gaussian_window_size, std=gaussian_std, sym=True)
    B = np.outer(gaussian_window, gaussian_window)
    
    n_win_ex = n_win + n_pad
    ph_bit = np.zeros((n_win_ex, n_win_ex), dtype=np.complex64)

    for ix1 in range(n_win_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + n_win
        
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win
            wf = np.vstack((np.zeros((i_shift, n_win)), 
                           wf[:n_win-i_shift, :]))

        for ix2 in range(n_win_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + n_win
            
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win
                wf2 = np.hstack((np.zeros((n_win, j_shift)), 
                                wf2[:, :n_win-j_shift]))

            ph_bit[:n_win, :n_win] = ph[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)
            # correct till here
            
            # Smooth response
            H = signal.convolve2d(np.fft.fftshift(H), B, mode='same') # values slightly different from MATLAB
            H = np.fft.ifftshift(H)
            
            meanH = np.median(H)
            if meanH != 0:
                H = H / meanH
                
            H = H ** alpha
            H = H - 1  # set all values under median to zero
            H[H < 0] = 0
            
            G = H * beta + low_pass
            ph_filt = np.fft.ifft2(ph_fft * G)
            ph_filt = ph_filt[:n_win, :n_win] * wf2
            
            if np.isnan(ph_filt[0, 0]):
                raise ValueError("NaN values detected in filtered phase")
                
            ph_out[i1:i2, j1:j2] += ph_filt

    return ph_out
