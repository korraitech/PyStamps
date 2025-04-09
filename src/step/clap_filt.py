import numpy as np
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
import warnings

def clap_filt(ph, alpha=0.5, beta=0.1, n_win=32, n_pad=0, low_pass=None):
    """
    Combined Low-pass Adaptive Phase filtering (Python translation of MATLAB code)

    Args:
        ph (np.ndarray): Input phase data (2D array). Assumed to be complex
                         if the operations involving FFT/IFFT are expected
                         to preserve complex values, or real otherwise.
                         NaNs will be replaced by 0.
        alpha (float, optional): Power for adaptive filter exponent. Defaults to 0.5.
        beta (float, optional): Scaling factor for adaptive filter. Defaults to 0.1.
        n_win (int, optional): Window size for processing blocks. Defaults to 32.
        n_pad (int, optional): Padding size added to the window for FFT. Defaults to 0.
        low_pass (np.ndarray, optional): Low-pass filter in the frequency domain
                                         (fftshifted). Must have dimensions
                                         (n_win + n_pad, n_win + n_pad).
                                         If None or empty, defaults to zeros.
                                         Defaults to None.

    Returns:
        np.ndarray: Filtered phase data, same size as input ph.
    """

    # Handle optional low_pass filter
    n_win_ex = n_win + n_pad # Extended window size including padding
    if low_pass is None:
        # Note: MATLAB's default is zeros(n_win+n_pad), which implies a square matrix
        low_pass = np.zeros((n_win_ex, n_win_ex), dtype=float) # Match MATLAB default type if needed
    elif low_pass.size == 0: # Check for empty array like MATLAB's isempty([])
         low_pass = np.zeros((n_win_ex, n_win_ex), dtype=float)
    elif low_pass.shape != (n_win_ex, n_win_ex):
         raise ValueError(f"low_pass filter must have shape ({n_win_ex}, {n_win_ex})")


    ph_out = np.zeros_like(ph, dtype=np.complex128) # Output should accommodate complex results from IFFT
    n_i, n_j = ph.shape

    # Ensure integer division where needed, consistent with MATLAB's floor/ceil
    n_inc = n_win // 4 # Use integer division, similar to MATLAB floor(n_win/4)
    if n_inc == 0:
        warnings.warn("n_win is too small (less than 4), setting n_inc to 1")
        n_inc = 1

    # Calculate number of windows needed, adjusting for MATLAB's calculation logic
    # MATLAB: ceil(n/n_inc) - 3 implies at least 4 windows are needed to start
    # Python equivalent needs careful calculation based on overlap
    n_win_i = int(np.ceil(n_i / n_inc)) - 3
    n_win_j = int(np.ceil(n_j / n_inc)) - 3

    # Check if dimensions are too small for the windowing scheme
    if n_win_i <= 0 or n_win_j <= 0:
        warnings.warn(f"Input dimensions ({n_i}, {n_j}) are too small for "
                      f"n_win={n_win} and n_inc={n_inc}. Result might be empty or incorrect. "
                      f"Consider reducing n_win or increasing input size.")
        # Handle gracefully: return empty or zeros if no processing happens
        if n_win_i <= 0 or n_win_j <= 0:
           return ph_out # Return the initialized zeros array


    # --- Create the spatial weighting window ('wind_func') ---
    # MATLAB: x = [0:n_win/2-1]; -> np.arange(n_win // 2)
    x = np.arange(n_win // 2)
    # MATLAB: [X,Y]=meshgrid(x,x);
    # Note: MATLAB's meshgrid default indexing is 'xy', NumPy's default is 'ij'
    #       To match MATLAB's output structure for X and Y:
    Y, X = np.meshgrid(x, x, indexing='ij') # Use 'ij' indexing for consistency
                                             # X corresponds to columns, Y to rows
    # MATLAB: X=X+Y;
    quadrant = X + Y  # Top-left quadrant
    # MATLAB: [X,fliplr(X)]
    top_half = np.hstack((quadrant, np.fliplr(quadrant)))
    # MATLAB: [wind_func;flipud(wind_func)];
    wind_func = np.vstack((top_half, np.flipud(top_half)))
    # MATLAB: wind_func=wind_func+1e-6;
    wind_func = wind_func + 1e-6 # Add small epsilon

    # --- Create the Gaussian filter kernel 'B' ---
    # MATLAB: gausswin(7) -> scipy.signal.windows.gaussian
    # MATLAB's gausswin(N) uses alpha=2.5 -> std = (N-1) / (2 * alpha)
    std_dev = (7 - 1) / (2 * 2.5) # std = 6 / 5 = 1.2
    gauss_win_1d = gaussian(7, std=std_dev, sym=True) # sym=True matches MATLAB default
    # MATLAB: gausswin(7)*gausswin(7)' -> Outer product
    B = np.outer(gauss_win_1d, gauss_win_1d)
    # Normalize B ? MATLAB's filter2 doesn't normalize implicitly, so we won't here unless needed.

    # --- Prepare input phase data ---
    ph_processed = np.copy(ph) # Work on a copy
    ph_processed[np.isnan(ph_processed)] = 0 # Replace NaNs with 0


    # --- Sliding window processing ---
    ph_bit = np.zeros((n_win_ex, n_win_ex), dtype=np.complex128) # Buffer for FFT

    # Loop over windows (adjusting for 0-based indexing)
    # MATLAB: for ix1=1:n_win_i -> range(n_win_i)
    for ix1 in range(n_win_i):
        wf = np.copy(wind_func) # Use a copy of the base window
        # MATLAB: i1=(ix1-1)*n_inc+1; -> ix1 * n_inc (0-based)
        i1 = ix1 * n_inc
        # MATLAB: i2=i1+n_win-1; -> i1 + n_win (exclusive end in Python slice)
        i2 = i1 + n_win

        # Boundary condition check (bottom edge)
        if i2 > n_i:
            i_shift = i2 - n_i # Amount overflowed
            i2 = n_i           # Clip end index
            i1 = n_i - n_win   # Adjust start index to maintain window size
            # Adjust window function (zero-pad top)
            # MATLAB: wf=[zeros(i_shift,n_win);wf(1:n_win-i_shift,:)];
            wf = np.vstack((np.zeros((i_shift, n_win)), wf[0:n_win - i_shift, :]))


        # MATLAB: for ix2=1:n_win_j -> range(n_win_j)
        for ix2 in range(n_win_j):
            wf2 = np.copy(wf) # Use a copy potentially modified by i-loop
            # MATLAB: j1=(ix2-1)*n_inc+1; -> ix2 * n_inc (0-based)
            j1 = ix2 * n_inc
            # MATLAB: j2=j1+n_win-1; -> j1 + n_win (exclusive end)
            j2 = j1 + n_win

            # Boundary condition check (right edge)
            if j2 > n_j:
               j_shift = j2 - n_j # Amount overflowed
               j2 = n_j           # Clip end index
               j1 = n_j - n_win   # Adjust start index
               # Adjust window function (zero-pad left)
               # MATLAB: wf2=[zeros(n_win,j_shift),wf2(:,1:n_win-j_shift)];
               wf2 = np.hstack((np.zeros((n_win, j_shift)), wf2[:, 0:n_win - j_shift]))


            # --- Core filtering operations for the current window ---
            # Extract data slice into padded buffer
            # MATLAB: ph_bit(1:n_win,1:n_win)=ph(i1:i2,j1:j2);
            # Python: Place the n_win x n_win slice into top-left of n_win_ex x n_win_ex buffer
            ph_bit.fill(0) # Clear buffer
            ph_bit[0:n_win, 0:n_win] = ph_processed[i1:i2, j1:j2]

            # FFT
            # MATLAB: ph_fft=fft2(ph_bit);
            ph_fft = np.fft.fft2(ph_bit)

            # Calculate adaptive filter component H
            # MATLAB: H=abs(ph_fft);
            H = np.abs(ph_fft)
            # MATLAB: H=ifftshift(filter2(B,fftshift(H)));
            # Note: filter2 in MATLAB performs convolution.
            #       scipy.signal.convolve2d is a good equivalent.
            #       'same' mode ensures output size matches input H.
            #       fftshift moves 0-freq to center for filtering, ifftshift moves it back.
            H_shifted = np.fft.fftshift(H)
            H_filtered_shifted = convolve2d(H_shifted, B, mode='same', boundary='fill', fillvalue=0)
            H = np.fft.ifftshift(H_filtered_shifted)

            # MATLAB: meanH=median(H(:));
            # Avoid RuntimeWarning for empty slice median with nanmedian
            # Use np.median as H should not contain NaNs here
            if H.size > 0:
                 # Use mean of non-zero elements if median is zero? No, MATLAB uses median directly.
                 meanH = np.median(H) # Median of all elements
            else:
                 meanH = 0 # Handle unlikely case of empty H

            # Normalize and apply exponent, handling potential division by zero
            if meanH != 0:
                H = H / meanH
            else:
                # If median is zero, H is likely all zeros or has issues.
                # Set H to zeros to avoid NaN/Inf from division/power.
                # Or maybe skip the adaptive part? MATLAB code implies H remains 0.
                H.fill(0) # If median is 0, likely H was mostly 0 anyway

            # MATLAB: H=H.^alpha;
            # Use np.power for complex numbers if H could become complex (unlikely here)
            # Since H = abs(fft), it starts real. filter2 keeps it real.
            # However, handle negative bases if alpha is not integer -> results would be complex/NaN
            # Since H = abs(...) / median(...), H should be >= 0.
            with np.errstate(invalid='ignore'): # Ignore warnings for 0^alpha if alpha<1
                H = np.power(H, alpha)
                H[np.isnan(H)] = 0 # Replace potential NaNs (e.g., 0^negative_alpha)


            # MATLAB: H=H-1; % set all values under median to zero
            H = H - 1
            # MATLAB: H(H<0)=0; % set all values under median to zero
            H[H < 0] = 0

            # Combine adaptive filter with low-pass filter
            # MATLAB: G=H*beta+low_pass;
            G = H * beta + low_pass # G is the combined filter in frequency domain

            # Apply filter and inverse FFT
            # MATLAB: ph_filt=ifft2(ph_fft.*G);
            ph_filt_padded = np.fft.ifft2(ph_fft * G)

            # Extract the relevant (non-padded) part and apply spatial window
            # MATLAB: ph_filt=ph_filt(1:n_win,1:n_win).*wf2;
            ph_filt = ph_filt_padded[0:n_win, 0:n_win] * wf2

            # Accumulate the result in the output array (weighted sum)
            # MATLAB: ph_out(i1:i2,j1:j2)=ph_out(i1:i2,j1:j2)+ph_filt;
            # Need to handle potential dtype mismatch if ph_out was not complex
            ph_out[i1:i2, j1:j2] = ph_out[i1:i2, j1:j2] + ph_filt

    # If the original input was real, the output ideally should be too,
    # but FFT/IFFT might introduce tiny imaginary parts.
    # Decide whether to return complex or real based on typical usage.
    # The MATLAB code doesn't explicitly take real(), so we return complex.
    # If input `ph` is known to be real and output should be real:
    # if np.isrealobj(ph):
    #    ph_out = np.real(ph_out)

    return ph_out