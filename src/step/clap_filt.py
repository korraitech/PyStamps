import numpy as np
from scipy.signal import convolve2d
from .utils import gaussian2D
import warnings

def clap_filt(ph, alpha=0.5, beta=0.1, n_win=32, n_pad=0, low_pass=None):

    ph_out = np.zeros_like(ph, dtype=np.complex128)
    n_i, n_j = ph.shape

    n_inc = n_win // 4
    if n_inc == 0:
        warnings.warn("n_win is too small (less than 4), setting n_inc to 1")
        n_inc = 1

    n_win_i = int(np.ceil(n_i / n_inc)) - 3
    n_win_j = int(np.ceil(n_j / n_inc)) - 3

    if n_win_i <= 0 or n_win_j <= 0:
        warnings.warn(f"Input dimensions ({n_i}, {n_j}) are too small for "
                      f"n_win={n_win} and n_inc={n_inc}. Result might be empty or incorrect. "
                      f"Consider reducing n_win or increasing input size.")
        if n_win_i <= 0 or n_win_j <= 0:
           return ph_out

    x = np.arange(n_win // 2)
    Y, X = np.meshgrid(x, x, indexing='ij')

    quadrant = X + Y
    top_half = np.hstack((quadrant, np.fliplr(quadrant)))
    wind_func = np.vstack((top_half, np.flipud(top_half)))
    wind_func = wind_func + 1e-6

    B = gaussian2D(7)

    ph_processed = np.copy(ph)
    ph_processed[np.isnan(ph_processed)] = 0

    n_win_ex = n_win + n_pad
    ph_bit = np.zeros((n_win_ex, n_win_ex), dtype=np.complex128)
    for ix1 in range(n_win_i):
        wf = np.copy(wind_func)
        i1 = ix1 * n_inc
        i2 = i1 + n_win

        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i          
            i1 = n_i - n_win  
            wf = np.vstack((np.zeros((i_shift, n_win)), wf[0:n_win - i_shift, :]))

        for ix2 in range(n_win_j):
            wf2 = np.copy(wf)
            j1 = ix2 * n_inc
            j2 = j1 + n_win

            if j2 > n_j:
               j_shift = j2 - n_j
               j2 = n_j
               j1 = n_j - n_win
               wf2 = np.hstack((np.zeros((n_win, j_shift)), wf2[:, 0:n_win - j_shift]))

            ph_bit.fill(0)
            ph_bit[0:n_win, 0:n_win] = ph_processed[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)
            H_shifted = np.fft.fftshift(H)
            H_filtered_shifted = convolve2d(H_shifted, B, mode='same', boundary='fill', fillvalue=0)
            H = np.fft.ifftshift(H_filtered_shifted)

            if H.size > 0:
                meanH = np.median(H)
            else:
                meanH = 0

            if meanH != 0:
                H = H / meanH
            else:
                H.fill(0)

            with np.errstate(invalid='ignore'):
                H = np.power(H, alpha)
                H[np.isnan(H)] = 0

            H = H - 1
            H[H < 0] = 0
            G = H * beta + low_pass
            ph_filt_padded = np.fft.ifft2(ph_fft * G)
            ph_filt = ph_filt_padded[0:n_win, 0:n_win] * wf2
            ph_out[i1:i2, j1:j2] = ph_out[i1:i2, j1:j2] + ph_filt

    return ph_out
