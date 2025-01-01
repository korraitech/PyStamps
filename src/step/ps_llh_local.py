import numpy as np

def ps_llh_local(llh, llh0):
    """
    Convert lat/lon/height coordinates to local coordinates.
    
    Parameters:
        llh: Array of [lon, lat, height] coordinates
        llh0: Reference [lon, lat] point
    
    Returns:
        xy: Local coordinates in kilometers
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis
    e = 0.08209443794970  # eccentricity (matching MATLAB version)
    
    # Extract coordinates
    lon = llh[0, :]
    lat = llh[1, :]
    
    # Convert to radians
    lon = np.radians(lon)
    lat = np.radians(lat)
    lon0 = np.radians(llh0[0])
    lat0 = np.radians(llh0[1])
    
    # Initialize output array
    xy = np.zeros((2, lon.shape[0]))
    
    # Handle non-zero latitudes
    z = lat != 0
    dlambda = lon[z] - lon0
    
    # Calculate meridian distance
    def meridian_dist(lat, a, e):
        return a * ((1-e**2/4-3*e**4/64-5*e**6/256)*lat - 
                   (3*e**2/8+3*e**4/32+45*e**6/1024)*np.sin(2*lat) +
                   (15*e**4/256 +45*e**6/1024)*np.sin(4*lat) -
                   (35*e**6/3072)*np.sin(6*lat))
    
    M = meridian_dist(lat[z], a, e)
    M0 = meridian_dist(lat0, a, e)
    
    # Calculate N and E
    N = a / np.sqrt(1 - e**2 * np.sin(lat[z])**2)
    E = dlambda * np.sin(lat[z])
    
    # Calculate local coordinates for non-zero latitudes
    xy[0, z] = N * 1 / np.tan(lat[z]) * np.sin(E)
    xy[1, z] = M - M0 + N * 1 / np.tan(lat[z]) * (1 - np.cos(E))
    
    # Handle zero latitudes
    nz = ~z
    dlambda = lon[nz] - lon0
    xy[0, nz] = a * dlambda
    xy[1, nz] = -M0
    
    # Convert to kilometers
    xy = xy / 1000
    
    return xy
