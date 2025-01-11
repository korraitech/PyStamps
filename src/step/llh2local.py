import numpy as np

def llh2local(llh, llh0):
    """
    Convert lat/lon/height coordinates to local coordinates.
    
    Parameters:
        llh: 2D array of shape (3, N), where rows are [lon, lat, height].
        llh0: 1D array-like of shape (2,), reference [lon, lat] point.
    
    Returns:
        xy: 2D array of shape (2, N) of local coordinates in kilometers.
    """
    # ----------------------------------------------------------------
    # Precompute constants for WGS84 ellipsoid and polynomial terms.
    # ----------------------------------------------------------------
    a = 6378137.0  # semi-major axis
    e = 0.08209443794970  # eccentricity (matching MATLAB version)

    e2 = e**2
    e4 = e2**2
    e6 = e4 * e2

    # Polynomial coefficients for the meridian distance expansion:
    c0 = 1 - (e2 / 4.0) - (3.0 * e4 / 64.0) - (5.0 * e6 / 256.0)
    c1 = (3.0 * e2 / 8.0) + (3.0 * e4 / 32.0) + (45.0 * e6 / 1024.0)
    c2 = (15.0 * e4 / 256.0) + (45.0 * e6 / 1024.0)
    c3 = (35.0 * e6 / 3072.0)

    def meridian_dist(lat):
        """
        Polynomial approximation for meridian distance on the WGS84 ellipsoid.
        """
        return a * (
            c0 * lat
            - c1 * np.sin(2.0 * lat)
            + c2 * np.sin(4.0 * lat)
            - c3 * np.sin(6.0 * lat)
        )

    # ----------------------------------------------------------------
    # Extract coordinates, convert to radians.
    # ----------------------------------------------------------------
    lon = np.radians(llh[0, :])
    lat = np.radians(llh[1, :])
    lon0 = np.radians(llh0[0])
    lat0 = np.radians(llh0[1])

    # ----------------------------------------------------------------
    # Allocate output array.
    # ----------------------------------------------------------------
    xy = np.zeros((2, lon.size))

    # ----------------------------------------------------------------
    # Identify lat != 0 for the main formula.
    # ----------------------------------------------------------------
    z = (lat != 0)
    nz = ~z  # lat == 0

    # ----------------------------------------------------------------
    # Meridian distances for offset computations, reference meridian distance.
    # ----------------------------------------------------------------
    M0 = meridian_dist(lat0)

    # ----------------------------------------------------------------
    # Handle points where lat != 0.
    # ----------------------------------------------------------------
    if np.any(z):
        lat_z = lat[z]
        dlambda_z = lon[z] - lon0

        # Precompute trigs once:
        sin_lat_z = np.sin(lat_z)
        tan_lat_z = np.tan(lat_z)

        # Meridian distances at lat != 0
        M = meridian_dist(lat_z)

        # Radius of curvature in the prime vertical:
        N = a / np.sqrt(1.0 - e2 * sin_lat_z**2)

        # E = dlambda * sin(lat)
        E = dlambda_z * sin_lat_z

        # x -> Easting
        xy[0, z] = N * (1.0 / tan_lat_z) * np.sin(E)
        # y -> Northing
        xy[1, z] = (M - M0) + N * (1.0 / tan_lat_z) * (1.0 - np.cos(E))

    # ----------------------------------------------------------------
    # Handle points where lat == 0 (purely equatorial).
    # ----------------------------------------------------------------
    if np.any(nz):
        dlambda_nz = lon[nz] - lon0
        xy[0, nz] = a * dlambda_nz
        xy[1, nz] = -M0

    # ----------------------------------------------------------------
    # Convert to kilometers and return.
    # ----------------------------------------------------------------
    xy /= 1000.0
    return xy
