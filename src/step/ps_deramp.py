import numpy as np

def ps_deramp(ps, ph_all):
    """
    Deramps the inputted data and gives that as output. Needs ps struct information!
    """

    print('Deramping computed on the fly.')

    n_ps = int(ps['n_ps'])
    
    # z = ax + by + c
    A = np.hstack((ps['xy'][:, 1:3] / 1000, np.ones((n_ps, 1))))

    ph_ramp = np.full(ph_all.shape, np.nan)
    for k in range(int(ps['n_ifg'])):
        ix = np.isnan(ph_all[:, k])
        if n_ps - np.sum(ix) > 5:
            coeff, _, _, _ = np.linalg.lstsq(A[~ix, :], ph_all[~ix, k], rcond=None)
            ph_ramp[:, k] = A @ coeff
            ph_all[:, k] = ph_all[:, k] - ph_ramp[:, k]
        else:
            print(f'Ifg {k} is not deramped')

    return ph_all, ph_ramp
