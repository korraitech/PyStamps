import numpy as np

def ps_deramp(ps, ph_all, degree=None):
    """
    Deramps the inputted data and gives that as output. Needs ps struct information!
    """

    print('Deramping computed on the fly.')

    if degree is None:
        degree = 1

    # SM from SB inversion deramping
    if ps['n_ifg'] != ph_all.shape[1]:
        ps['n_ifg'] = ph_all.shape[1]

    n_ps = int(ps['n_ps'])
    n_ifg = int(ps['n_ifg'])
    # detrending of the data
    if degree == 1:
        # z = ax + by + c
        A = np.hstack((ps['xy'][:, 1:3] / 1000, np.ones((n_ps, 1))))
        print('**** z = ax + by + c')
    elif degree == 1.5:
        # z = ax + by + cxy + d
        A = np.hstack((ps['xy'][:, 1:3] / 1000, (ps['xy'][:, 1] / 1000) * (ps['xy'][:, 2] / 1000), np.ones((n_ps, 1))))
        print('**** z = ax + by + cxy + d')
    elif degree == 2:
        # z = ax^2 + by^2 + cxy + d
        A = np.hstack(((ps['xy'][:, 1:3] / 1000) ** 2, (ps['xy'][:, 1] / 1000) * (ps['xy'][:, 2] / 1000), np.ones((n_ps, 1))))
        print('**** z = ax^2 + by^2 + cxy + d')
    elif degree == 3:
        # z = ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h
        A = np.hstack(((ps['xy'][:, 1:3] / 1000) ** 3, 
                       (ps['xy'][:, 1] / 1000) ** 2 * (ps['xy'][:, 2] / 1000), 
                       (ps['xy'][:, 2] / 1000) ** 2 * (ps['xy'][:, 1] / 1000),
                       (ps['xy'][:, 1:3] / 1000) ** 2, 
                       (ps['xy'][:, 1] / 1000) * (ps['xy'][:, 2] / 1000), 
                       np.ones((n_ps, 1))))
        print('**** z = ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h')

    ph_ramp = np.full(ph_all.shape, np.nan)
    for k in range(n_ifg):
        ix = np.isnan(ph_all[:, k])
        if n_ps - np.sum(ix) > 5:
            coeff, _, _, _ = np.linalg.lstsq(A[~ix, :], ph_all[~ix, k], rcond=None)
            ph_ramp[:, k] = A @ coeff
            ph_all[:, k] = ph_all[:, k] - ph_ramp[:, k]
        else:
            print(f'Ifg {k} is not deramped')

    return ph_all, ph_ramp
