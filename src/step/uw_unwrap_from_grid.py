import numpy as np
from scipy.io import loadmat

def uw_unwrap_from_grid(xy, pix_size):
    """
    Python translation of the MATLAB function uw_unwrap_from_grid.
    
    Attributes:
        xy (?): Unused in the MATLAB function shown.
        pix_size (?): Unused in the MATLAB function shown.
    
    Returns:
        ph_uw (ndarray): Unwrapped phase array, shape (n_ps, n_ifg).
        msd (float or ndarray): Value of 'msd' loaded from 'uw_phaseuw.mat'.
    """

    print("Unwrapping from grid...")

    # In MATLAB, load('uw_grid','nzix','n_ps','grid_ij','ph_in','ph_in_predef')
    # means load uw_grid.mat, specifically those variables. 
    # In Python, you get a dictionary. Adjust variable names if needed.
    uw = loadmat("uw_grid.mat")
    nzix          = uw["nzix"]          # shape and indexing may differ in Python
    n_ps          = int(uw["n_ps"].squeeze())  # might be a 1x1 array
    grid_ij       = uw["grid_ij"]       # 2D array
    ph_in         = uw["ph_in"]         # could be complex or real
    ph_in_predef  = uw["ph_in_predef"]  # can be empty or NaNs

    # Similarly, MATLAB load('uw_phaseuw') => load 'uw_phaseuw.mat'
    uu = loadmat("uw_phaseuw.mat")
    # The variable name "ph_uw" inside 'uw_phaseuw' might differ, 
    # so check your .mat file. Here we assume it's stored under "ph_uw" and "msd".
    ph_uw_loaded = uu["ph_uw"]  # shape (maybe n_ps x n_ifg)
    msd          = uu["msd"]

    # Determine array sizes
    # In MATLAB: [n_ps, n_ifg] = size(uw.ph_in); we'll replicate that in Python:
    n_ps_in, n_ifg = ph_in.shape

    # We create gridix with the same shape as nzix, then fill it:
    # In MATLAB: gridix=zeros(size(uw.nzix)); gridix(uw.nzix)=[1:uw.n_ps];
    # But Python is 0-based indexing, so we have to be careful.
    gridix = np.zeros_like(nzix, dtype=int)

    # This line in MATLAB sets gridix(:) at indices given by nzix(:) to 1..n_ps
    # If nzix is a mask of some sort, you may need to unravel it or fix indexing.
    # Here we assume flat indexing. Confirm shape and indexing in your data.
    gridix_flat = gridix.ravel()
    nzix_flat = nzix.ravel().astype(int)  # convert to int
    valid_mask = (nzix_flat > 0)  # which elements are > 0?
    # In MATLAB:  gridix(uw.nzix) = 1:n_ps
    # implies the first nonzero place in nzix -> 1, second -> 2, etc.
    # We'll do something analogous:
    indices = np.nonzero(valid_mask)[0]
    # Fill in 1..n_ps sequentially (MATLAB-likes do 1 through n_ps):
    for idx, ps_idx in zip(indices, range(1, n_ps+1)):
        gridix_flat[idx] = ps_idx

    # reshape back
    gridix = gridix_flat.reshape(gridix.shape)

    # Prepare output array
    ph_uw = np.full((n_ps, n_ifg), np.nan, dtype=np.float32)

    # For i in 1..n_ps (MATLAB), in Python it’s 0..(n_ps-1)
    for i in range(n_ps):
        # grid_ij(i, 1), grid_ij(i, 2) in MATLAB => in Python, that’s grid_ij[i, 0], grid_ij[i, 1] (0-based).
        # But watch out – if your MATLAB grid_ij is 1-based indexing, 
        # you might have to subtract 1 for Python indexing. 
        # We'll assume your loaded grid_ij is 0-based or already matches Python indexing. 
        r, c = grid_ij[i, 0], grid_ij[i, 1]

        # 'ix' is 1-based in MATLAB, so might need to subtract 1 for Python indexing.
        ix = gridix[r, c]
        
        if ix == 0:
            # Same logic as MATLAB: ph_uw(i,:) = nan
            ph_uw[i, :] = np.nan
        else:
            # In MATLAB: ph_uw_pix = uu.ph_uw(ix, :)
            # Because ix in MATLAB is 1-based. So in Python, it might be ix-1.
            ph_uw_pix = ph_uw_loaded[ix-1, :]  # if your data is 1-based
            # If ph_in is real or complex
            if np.isrealobj(ph_in):
                # ph_uw(i,:) = ph_uw_pix + angle(exp(1i*(ph_in(i,:)-ph_uw_pix)));
                diff = ph_in[i, :] - ph_uw_pix
                ph_uw[i, :] = ph_uw_pix + np.angle(np.exp(1j * diff))
            else:
                # ph_uw(i,:) = ph_uw_pix + angle(ph_in(i,:).*exp(-1i*ph_uw_pix));
                diff = ph_in[i, :] * np.exp(-1j * ph_uw_pix)
                ph_uw[i, :] = ph_uw_pix + np.angle(diff)

    # Now handle the "ph_in_predef" part
    if ph_in_predef.size > 0:
        # In MATLAB: predef_ix=~isnan(uw.ph_in_predef);
        predef_ix = ~np.isnan(ph_in_predef)
        # meandiff=nanmean(ph_uw-uw.ph_in_predef);
        # meandiff=2*pi*round(meandiff/2/pi);
        diff = ph_uw - ph_in_predef
        meandiff = np.nanmean(diff, axis=0)
        meandiff = 2 * np.pi * np.round(meandiff / (2 * np.pi))

        # uw.ph_in_predef=uw.ph_in_predef+repmat(meandiff,n_ps,1);
        # We'll just broadcast meandiff since it's shape (n_ifg,).
        ph_in_predef += meandiff

        # ph_uw(predef_ix)=uw.ph_in_predef(predef_ix);
        ph_uw[predef_ix] = ph_in_predef[predef_ix]

    return ph_uw, msd