import numpy as np
from .utils import gaussian2D,save_h5
from scipy.signal import convolve2d

def wrap_filt(ph, n_win, alpha, low_flag):
    
    n_pad = round(n_win * 0.25)
    
    n_i, n_j = ph.shape
    n_inc = n_win // 2
    n_win_i = int(np.ceil(n_i / n_inc)) - 1
    n_win_j = int(np.ceil(n_j / n_inc)) - 1
    
    ph_out = np.zeros_like(ph)
    if low_flag.lower() == 'y':
        ph_out_low = np.zeros_like(ph)
    else:
        ph_out_low = []
    
    x = np.arange(1, n_win//2 + 1)
    X, Y = np.meshgrid(x, x)
    X = X + Y
    wind_func = np.vstack([
        np.hstack([X, np.fliplr(X)]),
        np.flipud(np.hstack([X, np.fliplr(X)]))
    ])
    

    ph = np.nan_to_num(ph)
    B = gaussian2D(7)
    ph_bit = np.zeros((n_win + n_pad, n_win + n_pad), dtype=np.complex64)
    L = np.fft.ifftshift(gaussian2D(n_win + n_pad, 16))
    
    for ix1 in range(n_win_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + n_win
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win
            wf = np.vstack([
                np.zeros((i_shift, n_win)),
                wf[:n_win-i_shift, :]
            ])
            
        for ix2 in range(n_win_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + n_win
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win
                wf2 = np.hstack([
                    np.zeros((n_win, j_shift)),
                    wf2[:, :n_win-j_shift]
                ])
            
            ph_bit[:n_win, :n_win] = ph[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)
            H = np.fft.ifftshift(convolve2d(np.fft.fftshift(H),B,mode='same'))
            
            meanH = np.median(H)
            if meanH != 0:
                H = H / meanH
            
            H = H ** alpha
            ph_filt = np.fft.ifft2(ph_fft * H)
            ph_filt = ph_filt[:n_win, :n_win] * wf2
            
            if low_flag.lower() == 'y':
                ph_filt_low = np.fft.ifft2(ph_fft * L)
                ph_filt_low = ph_filt_low[:n_win, :n_win] * wf2
            
            ph_out[i1:i2, j1:j2] += ph_filt
            if low_flag.lower() == 'y':
                ph_out_low[i1:i2, j1:j2] += ph_filt_low
    
    # Reset magnitude
    ph_out = np.abs(ph) * np.exp(1j * np.angle(ph_out))
    if low_flag.lower() == 'y':
        ph_out_low = np.abs(ph) * np.exp(1j * np.angle(ph_out_low))
    
    return ph_out, ph_out_low 

def uw_grid_wrapped(workdir, ph_in, xy_in, options):

    print('Resampling phase to grid...')

    pix_size = options['grid_size']
    prefilt_win = options['prefilt_win']
    lowfilt_flag = options['lowfilt_flag']
    goldfilt_flag = options['goldfilt_flag']
    gold_alpha = options['gold_alpha']

    n_ps, n_ifg = ph_in.shape
    print(f'   Number of interferograms  : {n_ifg}')
    print(f'   Number of points per ifg  : {n_ps}')

    grid_x_min = np.min(xy_in[:, 1])
    grid_y_min = np.min(xy_in[:, 2])

    # Calculate grid indices
    grid_ij = np.zeros((n_ps, 2), dtype=int)
    grid_ij[:, 0] = np.ceil((xy_in[:, 2] - grid_y_min + 1e-3) / pix_size).astype(int)
    grid_ij[:, 1] = np.ceil((xy_in[:, 1] - grid_x_min + 1e-3) / pix_size).astype(int)

    # Adjust maximum indices
    grid_ij[grid_ij[:, 0] == np.max(grid_ij[:, 0]), 0] = np.max(grid_ij[:, 0]) - 1
    grid_ij[grid_ij[:, 1] == np.max(grid_ij[:, 1]), 1] = np.max(grid_ij[:, 1]) - 1

    n_i = np.max(grid_ij[:, 0])
    n_j = np.max(grid_ij[:, 1])

    ph_grid = np.zeros((n_i, n_j), dtype=np.complex64)

    if min(n_i, n_j) < prefilt_win:
        raise ValueError(f'Minimum dimension of the resampled grid ({min(n_i, n_j)} pixels) '
                        f'is less than prefilter window size ({prefilt_win})')
    for i1 in range(n_ifg):
        if np.isreal(ph_in).all():
            ph_this = np.exp(1j * ph_in[:, i1])
        else:
            ph_this = ph_in[:, i1]

        ph_grid.fill(0)
        for i in range(n_ps):
            ph_grid[grid_ij[i, 0]-1, grid_ij[i, 1]-1] += ph_this[i]

        if i1 == 0:
            nzix = ph_grid != 0
            n_ps_grid = np.sum(nzix)
            ph = np.zeros((n_ps_grid, n_ifg), dtype=np.complex64)
            if lowfilt_flag.lower() == 'y':
                ph_lowpass = np.zeros_like(ph)
            else:
                ph_lowpass = []

        if goldfilt_flag.lower() == 'y' or lowfilt_flag.lower() == 'y':
            ph_this_gold, ph_this_low = wrap_filt(ph_grid, prefilt_win, gold_alpha, lowfilt_flag)
            if lowfilt_flag.lower() == 'y':
                ph_lowpass[:, i1] = ph_this_low[nzix]
        
        if goldfilt_flag.lower() == 'y':
            ph_flat = ph_this_gold.flatten(order='F')
            nzix_flat = nzix.flatten(order='F')
            ph[:, i1] = ph_flat[nzix_flat]
        else:
            ph[:, i1] = ph_grid[nzix]

    n_ps = n_ps_grid
    print(f'   Number of resampled points: {n_ps}')

    # Find non-zero indices
    nz_i, nz_j = np.where(ph_grid != 0)
    xy = np.column_stack(((nz_j+ 0.5) * pix_size, (nz_i + 0.5) * pix_size))
    xy = np.column_stack((np.arange(n_ps),xy[np.lexsort((xy[:, 1], xy[:, 0]))]))
    ij = np.column_stack((nz_i+ 1, nz_j + 1))
    ij = ij[np.lexsort((ij[:, 0], ij[:, 1]))]

    save_h5(workdir,'uw_grid.h5',**{'ph':ph, 'ph_in':ph_in, 'ph_lowpass':ph_lowpass,
                                    'xy':xy, 'ij':ij, 'nzix':nzix, 'grid_x_min':grid_x_min,
                                    'grid_y_min':grid_y_min, 'n_i':n_i, 'n_j':n_j, 'n_ifg':n_ifg,
                                    'n_ps':n_ps, 'grid_ij':grid_ij, 'pix_size':pix_size})
