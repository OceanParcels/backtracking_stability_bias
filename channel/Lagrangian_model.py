import numpy as np
import xarray
import math
import tqdm
import xarray as xr
from collections import deque
from numba import jit, njit, prange

"""
This is bespoke code for Lagrangian trajectory computations
in the MITgcm flat channel model. Some of the code is specific
to the model output format, but the general structure should
be useful for other applications.
This code includes functions to interpolate model output and
to compute trajectories using Runge-Kutta and the Analytical method.

TODO
 - Implement three-dimensional advection
"""

# Here are some hardcoded model parameters
dx = 5000
dy = 5000
xidx_max = 200
yidx_max = 400

# MITgcm cell thicknesses
dz_arr = np.array([  5.4871655,   6.194621 ,   6.9929113,   7.8935375,   8.909376 ,
                10.054832 ,  11.345955 ,  12.800571 ,  14.438377 ,  16.281029 ,
                18.352104 ,  20.67704  ,  23.282867 ,  26.197693 ,  29.450119 ,
                33.067932 ,  37.076553 ,  41.496918 ,  46.34247  ,  51.615936 ,
                57.305176 ,  63.37964  ,  69.78656  ,  76.45001  ,  83.27045  ,
                90.130005 ,  96.89899  , 103.44629  , 109.651    , 115.41223  ,
                120.65698  , 125.342896 , 129.45825  , 133.01648  , 136.05078  ,
                138.60803  , 140.74072  , 142.5044   , 143.95215  , 145.13379  ,
                146.09302  , 146.86938  , 147.49463  , 147.9978   , 148.40137  ,
                148.72437  , 148.98315  , 149.1897   , 149.35449  , 149.35449  ],
                dtype=np.float64)

# MITgcm cell walls
Zl = np.array([    0.      ,    -5.487165,   -11.681787,   -18.674698,   -26.568235,
                -35.47761 ,   -45.532444,   -56.8784  ,   -69.67897 ,   -84.11735 ,
                -100.39838 ,  -118.75048 ,  -139.42752 ,  -162.71039 ,  -188.90808 ,
                -218.3582  ,  -251.42613 ,  -288.5027  ,  -329.9996  ,  -376.34207 ,
                -427.958   ,  -485.26318 ,  -548.6428  ,  -618.4294  ,  -694.8794  ,
                -778.14984 ,  -868.27985 ,  -965.17883 , -1068.6251  , -1178.2761  ,
                -1293.6884  , -1414.3453  , -1539.6882  , -1669.1465  , -1802.163   ,
                -1938.2137  , -2076.8218  , -2217.5625  , -2360.067   , -2504.019   ,
                -2649.1528  , -2795.2458  , -2942.1152  , -3089.6099  , -3237.6077  ,
                -3386.009   , -3534.7334  , -3683.7166  , -3832.9062  ], 
                dtype=np.float64)


class pset:
    """
    Set of particles. Convenient class for analysis.
    """
    def __init__(self, x_init, y_init, z_init = None):
        assert x_init.shape == y_init.shape, "All initial position arrays must be the same shape."
        if z_init is not None:
            assert x_init.shape == z_init.shape, "All initial position arrays must be the same shape."
        assert len(x_init.shape) == 1, "Only 1-D input arrays allowed."
        self.x_init = x_init
        self.y_init = y_init
        self.z_init = z_init
        self.metadata = {}

    @property
    def size(self):
        return self.x_init.size

    def store_results(self, X, Y, Z=None, T=None):
        """
        Store simulation results in the object.

        Parameters
        ----------
        X : numpy.ndarray
            Array of x-coordinates over time.
        Y : numpy.ndarray
            Array of y-coordinates over time.
        Z : numpy.ndarray, optional
            Array of z-coordinates over time, if applicable.
        T : numpy.ndarray, optional
            Array of time steps at which coordinates were recorded.
        """
        assert X.shape == Y.shape, "Shapes of output should all be the same"
        if Z is not None:
            assert X.shape == Z.shape, "Shapes of output should all be the same"
        assert X.shape[1] == self.x_init.size, "Results should have the same size as initial positions."
        self.X = X
        self.Y = Y
        self.Z = Z
        self.T = T
    
    @property
    def T_total(self):
        return np.abs(T[-1] - T[0])

    @classmethod
    def from_forward(cls, partSet):
        x_init = partSet.X[-1]
        y_init = partSet.Y[-1]
        if partSet.Z is not None:
            z_init = partSet.Z[-1]
        else:
            z_init=None
        return cls(x_init, y_init, z_init)

    def add_metadata(self, key, value):
        self.metadata[key] = value

    def to_netcdf(self, filename):
        """
        Saves the particle positions and times to a NetCDF file.

        Parameters
        ----------
        filename : str
            Path to the file where the data will be saved.
        """
        data_vars = {
            'X': (('obs', 'trajectory'), self.X),
            'Y': (('obs', 'trajectory'), self.Y),
            'T': ('obs', self.T)
        }
        if self.Z is not None:
            data_vars['Z'] = (('obs', 'trajectory'), self.Z)
        
        ds = xr.Dataset(data_vars)

        encoding = {var: {"zlib": True, "complevel": 2} for var in ['X', 'Y']}
        if self.Z is not None:
            encoding['Z'] = {"zlib": True, "complevel": 2}

        if filename[-3:] != '.nc':
            filename += '.nc'

        ds.attrs = self.metadata
        ds.to_netcdf(filename, encoding=encoding)
        
        print(f"Written to: ", filename)

    @classmethod
    def from_netcdf(cls, filename):
        """
        Creates a pset instance from a NetCDF file.

        Parameters
        ----------
        filename : str
            Path to the NetCDF file to load.

        Returns
        -------
        pset
            A new pset instance with data loaded from the NetCDF file.
        """
        ds = xr.open_dataset(filename)
        X = ds['X'].values
        Y = ds['Y'].values
        T = ds['T'].values
        Z = ds['Z'].values if 'Z' in ds else None

        # Infer initial positions as the first record of the data
        x_init = X[0, :]
        y_init = Y[0, :]
        z_init = Z[0, :] if Z is not None else None

        instance = cls(x_init, y_init, z_init)
        instance.store_results(X, Y, Z, T)
        instance.metadata = ds.attrs
        return instance


def time_interpolate(ds, t_sec, ds_dt = 24 * 60 * 60):
    """
    Interpolate a dataset to the time t_sec (seconds since the start of the dataset).

    NOTE: `time_interpolate_numpy` is faster. 

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to interpolate
    t_sec : int
        The time in seconds to interpolate to
    ds_dt : int
        The output dt in seconds of the dataset
    """
    t_left_idx = int(t_sec // ds_dt)
    t_right_idx = int(t_left_idx + 1)

    ds_left = ds.isel(time=t_left_idx)
    ds_right = ds.isel(time=t_right_idx)

    ds_interp = ds_left + (ds_right - ds_left) * (t_sec % ds_dt) / ds_dt

    return ds_interp


@njit
def time_interpolate_numpy(UVEL, VVEL, t_sec, ds_dt = 24 * 60 * 60):
    """
    Interpolate a dataset to the time t_sec (seconds since the start of the dataset)
    Uses numpy for faster interpolation.

    Parameters
    ----------
    UVEL : np.ndarray
        Zonal velocities to interpolate
    VVEL : np.ndarray
        Meridional velocities to interpolate
    t_sec : int
        The time in seconds to interpolate to
    ds_dt : int
        The output dt in seconds of the dataset

    Returns
    -------
    UVEL_interp : np.array
        Interpolated U-velocity
    VVEL_interp : np.array
        Interpolated V-velocity
    """
    t_left_idx = round(t_sec // ds_dt)
    t_right_idx = round(t_left_idx + 1)

    UVEL_left = UVEL[t_left_idx]
    UVEL_right = UVEL[t_right_idx]

    VVEL_left = VVEL[t_left_idx]
    VVEL_right = VVEL[t_right_idx]

    UVEL_interp = UVEL_left + (UVEL_right - UVEL_left) * (t_sec % ds_dt) / ds_dt
    VVEL_interp = VVEL_left + (VVEL_right - VVEL_left) * (t_sec % ds_dt) / ds_dt

    return UVEL_interp, VVEL_interp


@njit
def time_select_numpy(UVEL, VVEL, t_sec, ds_dt = 24 * 60 * 60, dt=1):
    """
    Interpolate a dataset to the time t_sec (seconds since the start of the dataset)
    Uses numpy for faster interpolation.

    Parameters
    ----------
    UVEL : np.ndarray
        Zonal velocities to interpolate
    VVEL : np.ndarray
        Meridional velocities to interpolate
    t_sec : int
        The time in seconds to interpolate to
    ds_dt : int
        The output dt in seconds of the dataset
    direction : int
        Dt used for the simulation. Used for infering direction.

    Returns
    -------
    UVEL_select : np.array
        Interpolated U-velocity
    VVEL_select : np.array
        Interpolated V-velocity
    """
    if t_sec % ds_dt == 0:
        t_idx = round(t_sec // ds_dt) + int(np.sign(dt) / 2 - 0.5)
    else:
        t_idx = round(t_sec // ds_dt)

    UVEL_select = UVEL[t_idx]
    VVEL_select = VVEL[t_idx]

    return UVEL_select, VVEL_select

@njit
def time_interpolate_numpy_3D(UVEL, VVEL, WVEL, t_sec, ds_dt = 24 * 60 * 60):
    """
    Interpolate a dataset to the time t_sec (seconds since the start of the dataset)
    Uses numpy for faster interpolation.

    Parameters
    ----------
    UVEL : np.ndarray
        Zonal velocities to interpolate
    VVEL : np.ndarray
        Meridional velocities to interpolate
    WVEL : np.ndarray
        Vertical velocities to interpolate
    t_sec : int
        The time in seconds to interpolate to
    ds_dt : int
        The output dt in seconds of the dataset

    Returns
    -------
    UVEL_interp : np.array
        Interpolated U-velocity
    VVEL_interp : np.array
        Interpolated V-velocity
    WVEL_interp : np.array
        Interpolated W-velocity
    """
    t_left_idx = round(t_sec // ds_dt)
    t_right_idx = round(t_left_idx + 1)

    UVEL_left = UVEL[t_left_idx]
    UVEL_right = UVEL[t_right_idx]

    VVEL_left = VVEL[t_left_idx]
    VVEL_right = VVEL[t_right_idx]

    WVEL_left = WVEL[t_left_idx]
    WVEL_right = WVEL[t_right_idx]

    UVEL_interp = UVEL_left + (UVEL_right - UVEL_left) * (t_sec % ds_dt) / ds_dt
    VVEL_interp = VVEL_left + (VVEL_right - VVEL_left) * (t_sec % ds_dt) / ds_dt
    WVEL_interp = WVEL_left + (WVEL_right - WVEL_left) * (t_sec % ds_dt) / ds_dt

    return UVEL_interp, VVEL_interp, WVEL_interp


@njit
def find_G_idx(X, Y):
    """
    Given a point (X, Y), find the index of the cell's lower left F-point. 
    From this, we can find the indices of the U and V points. 

    Parameters
    ----------
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point

    Returns
    -------
    G_x_idx : int; numpy array
        x-index of the lower left G-point
    G_y_idx : int; numpy array
        y-index of the lower left G-point
    """

    G_x_idx = (np.floor(X / dx)).astype(np.int32)
    G_y_idx = (np.floor(Y / dy)).astype(np.int32)

    return G_x_idx, G_y_idx


@njit
def find_G_idx_3D(X, Y, Z):
    """
    Given a point (X, Y), find the index of the cell's lower left F-point. 
    From this, we can find the indices of the U and V points. 

    Parameters
    ----------
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point
    Z : float; numpy array
        z-coordinate of the point

    Returns
    -------
    XG_idx : int; numpy array
        x-index of the lower left G-point
    YG_idx : int; numpy array
        y-index of the lower left G-point
    """

    XG_idx = (np.floor(X / dx)).astype(np.int32)
    YG_idx = (np.floor(Y / dy)).astype(np.int32)
    ZG_idx = (np.searchsorted(-Zl, -Z, side='right') - 1).astype(np.int32)    # Hardcoded for MITgcm

    return XG_idx, YG_idx, ZG_idx


def UV_interp(ds, X, Y):
    """
    Interpolate the velocity field to a point (X, Y), using C-grid interpolation.

    Parameters
    ----------
    ds : xarray dataset
        Dataset containing the velocity fields
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point

    Returns
    -------
    U_interp : np.array, float
        Interpolated U-velocity per point
    V_interp : np.array, float
        Interpolated V-velocity per point
    """

    X_idx, Y_idx = find_G_idx(X, Y)

    U0 = ds.UVEL.values[Y_idx % yidx_max, X_idx % xidx_max]
    U1 = ds.UVEL.values[Y_idx % yidx_max, (X_idx + 1) % xidx_max]
    V0 = ds.VVEL.values[Y_idx % yidx_max, X_idx % xidx_max]
    V1 = ds.VVEL.values[(Y_idx + 1) % yidx_max, X_idx % xidx_max]

    U_interp = U0 + (X/dx - X_idx) * (U1 - U0)
    V_interp = V0 + (Y/dy - Y_idx) * (V1 - V0)

    return U_interp, V_interp


@njit
def get_elements_by_indices_2D(data, row_indices, col_indices):
    """
    Retrieve elements from a 2D array using arrays of indices.

    Parameters
    ----------
    data : numpy.ndarray
        The array (2D) from which to retrieve elements.
    row_indices : numpy.ndarray
        An array of row indices.
    col_indices : numpy.ndarray
        An array of column indices.
    depth_indices : numpy.ndarray, optional

    Returns
    -------
    numpy.ndarray: An array of elements corresponding to the provided indices.
    """
    assert len(row_indices) == len(col_indices), "Row and column indices must have the same length"
    result = np.empty(len(row_indices), dtype=data.dtype)
    for i in range(len(row_indices)):
        result[i] = data[row_indices[i], col_indices[i]]
    return result


@njit
def get_elements_by_indices_3D(data, row_indices, col_indices, depth_indices):
    """
    Retrieve elements from a 3D array using arrays of indices.
    NOTE: Depth should be interpreted in the array-sense.

    Parameters
    ----------
    data : numpy.ndarray
        The array (3D) from which to retrieve elements.
    row_indices : numpy.ndarray
        An array of row indices.
    col_indices : numpy.ndarray
        An array of column indices.
    depth_indices : numpy.ndarray, optional
        An optional array of depth indices for 3D arrays.

    Returns
    -------
    numpy.ndarray: An array of elements corresponding to the provided indices.
    """
    assert len(row_indices) == len(col_indices) == len(depth_indices), "All indices must have the same length"
    result = np.empty(len(row_indices), dtype=data.dtype)
    for i in range(len(row_indices)):
        result[i] = data[row_indices[i], col_indices[i], depth_indices[i]]
    return result


@njit
def UV_interp_numpy(U_field, V_field, X, Y):
    """
    Interpolate the velocity field to a point (X, Y), using C-grid interpolation.

    Parameters
    ----------
    U_field : numpy array
        x-velocity field
    V_field : numpy array
        y-velocity field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point

    Returns:
    U_interp : np.array, float
        Interpolated U-velocity per point
    V_interp: np.array, float
        Interpolated V-velocity per point
    """
    X_idx, Y_idx = find_G_idx(X, Y)

    # U0 = U_field[Y_idx % yidx_max, X_idx % xidx_max]
    # U1 = U_field[Y_idx % yidx_max, (X_idx + 1) % xidx_max]
    # V0 = V_field[Y_idx % yidx_max, X_idx % xidx_max]
    # V1 = V_field[(Y_idx + 1) % yidx_max, X_idx % xidx_max]

    # Use `get_elements_by_indices` function to be numba-compatible
    U0 = get_elements_by_indices_2D(U_field, Y_idx % yidx_max, X_idx % xidx_max)
    U1 = get_elements_by_indices_2D(U_field, Y_idx % yidx_max, (X_idx + 1) % xidx_max)
    V0 = get_elements_by_indices_2D(V_field, Y_idx % yidx_max, X_idx % xidx_max)
    V1 = get_elements_by_indices_2D(V_field, (Y_idx + 1) % yidx_max, X_idx % xidx_max)

    U_interp = U0 + (X/dx - X_idx) * (U1 - U0)
    V_interp = V0 + (Y/dy - Y_idx) * (V1 - V0)

    return U_interp, V_interp

@njit
def F_nearest_interp_numpy(F_field, X, Y):
    """
    Interpolate a F-point field to a point (X, Y), using nearest.

    Parameters
    ----------
    T_field : numpy array
        tracer field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point

    Returns:
    T_interp: np.array, float
        Interpolated T-values per point
    """
    # this function finds the lower left point, so we have to shift
    X_idx, Y_idx = find_G_idx(X + dx/2, Y + dx/2) 

    # Use `get_elements_by_indices` function to be numba-compatible
    F_interp = get_elements_by_indices_2D(F_field, Y_idx % yidx_max, X_idx % xidx_max)
    
    return F_interp

@njit
def U_nearest_interp_numpy(U_field, X, Y):
    """
    Interpolate a U-point field to a point (X, Y), using nearest.

    Parameters
    ----------
    T_field : numpy array
        tracer field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point

    Returns:
    T_interp: np.array, float
        Interpolated T-values per point
    """
    # this function finds the lower left point. The U-point lives in the middle left
    # so we have to shift Y
    X_idx, Y_idx = find_G_idx(X, Y + dy/2)

    # Use `get_elements_by_indices` function to be numba-compatible
    U_interp = get_elements_by_indices_2D(T_field, Y_idx % yidx_max, X_idx % xidx_max)
    
    return U_interp


@njit
def V_nearest_interp_numpy(V_field, X, Y):
    """
    Interpolate a V-point field to a point (X, Y), using nearest.

    Parameters
    ----------
    T_field : numpy array
        tracer field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point

    Returns:
    T_interp: np.array, float
        Interpolated T-values per point
    """
    # this function finds the lower left point. The U-point lives in the lower middle
    # so we have to shift X
    X_idx, Y_idx = find_G_idx(X + dx/2, Y)

    # Use `get_elements_by_indices` function to be numba-compatible
    V_interp = get_elements_by_indices_2D(T_field, Y_idx % yidx_max, X_idx % xidx_max)
    
    return V_interp


@njit
def T_nearest_interp_numpy(T_field, X, Y):
    """
    Interpolate a T-point field to a point (X, Y), using nearest.

    Parameters
    ----------
    T_field : numpy array
        tracer field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point

    Returns:
    T_interp: np.array, float
        Interpolated T-values per point
    """
    X_idx, Y_idx = find_G_idx(X, Y)

    # Use `get_elements_by_indices` function to be numba-compatible
    T_interp = get_elements_by_indices_2D(T_field, Y_idx % yidx_max, X_idx % xidx_max)
    
    return T_interp

@njit
def T_linear_interp_numpy(T_field, X, Y):
    """
    Interpolate a T-point field to a point (X, Y), using bilinear interpolation.

    Parameters
    ----------
    T_field : numpy array
        tracer field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point

    Returns:
    T_interp: np.array, float
        Interpolated T-values per point
    """
    # Find indices of the lower left T-point
    X_idx, Y_idx = find_G_idx(X + dx/2, Y + dy/2) # since the tracer is located at the T-points

    # Use `get_elements_by_indices` function to be numba-compatible
    T_interp = get_elements_by_indices_2D(T_field, Y_idx % yidx_max, X_idx % xidx_max)

    T00 = get_elements_by_indices_2D(T_field, Y_idx % yidx_max, X_idx % xidx_max)
    T01 = get_elements_by_indices_2D(T_field, Y_idx % yidx_max, (X_idx + 1) % xidx_max)
    T10 = get_elements_by_indices_2D(T_field, (Y_idx + 1) % yidx_max, X_idx % xidx_max)
    T11 = get_elements_by_indices_2D(T_field, (Y_idx + 1) % yidx_max, (X_idx + 1) % xidx_max)

    T_interp = (
        T00 * ((X_idx + 1.5 * dx) - X) * ((Y_idx + 1.5 * dy) - Y) +
        T10 * (X - (X_idx + 0.5 * dx)) * ((Y_idx + 1.5 * dy) - Y) +
        T01 * ((X_idx + 1.5 * dx) - X) * (Y - (Y_idx + 0.5 * dy)) +
        T11 * (X - (X_idx + 0.5 * dx)) * (Y - (Y_idx + 0.5 * dy))
    ) / (dx * dy)
    
    return T_interp


def T_nearest_interp_numpy_nojit(T_field, X, Y):
    """
    Interpolate a T-point field to a point (X, Y), using nearest.

    Parameters
    ----------
    T_field : numpy array
        tracer field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point

    Returns:
    T_interp: np.array, float
        Interpolated T-values per point
    """
    X_idx, Y_idx = find_G_idx(X, Y)

    T_interp = get_elements_by_indices_2D(T_field, Y_idx % yidx_max, X_idx % xidx_max)
    # T_interp = T_field[Y_idx % yidx_max, X_idx % xidx_max]

    return T_interp

@njit
def UVW_interp_numpy(U_field, V_field, W_field, X, Y, Z):
    """
    Interpolate the velocity field to a point (X, Y, Z), using C-grid interpolation.

    Parameters
    ----------
    U_field : numpy array
        x-velocity field
    V_field : numpy array
        y-velocity field
    W_field : numpy array
        z-velocity field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point
    Z : float; numpy array
        z-coordinate of the point

    Returns:
    U_interp : np.array, float
        Interpolated U-velocity per point
    V_interp: np.array, float
        Interpolated V-velocity per point
    W_interp : np.array, float
        Interpolated W-velocity per point
    """
    X_idx, Y_idx, Z_idx = find_G_idx_3D(X, Y, Z)

    dz = dz_arr[Z_idx].astype(np.float64)

    # U0 = U_field[Z_idx, Y_idx % yidx_max, X_idx % xidx_max]
    # U1 = U_field[Z_idx, Y_idx % yidx_max, (X_idx + 1) % xidx_max]
    # V0 = V_field[Z_idx, Y_idx % yidx_max, X_idx % xidx_max]
    # V1 = V_field[Z_idx, (Y_idx + 1) % yidx_max, X_idx % xidx_max]
    # W0 = W_field[Z_idx, Y_idx % yidx_max, X_idx % xidx_max]
    # W1 = W_field[Z_idx + 1, Y_idx % yidx_max, X_idx % xidx_max]

    # Use `get_elements_by_indices` function to be numba-compatible
    U0 = get_elements_by_indices_3D(U_field, Z_idx, Y_idx % yidx_max, X_idx % xidx_max)
    U1 = get_elements_by_indices_3D(U_field, Z_idx, Y_idx % yidx_max, (X_idx + 1) % xidx_max)
    V0 = get_elements_by_indices_3D(V_field, Z_idx, Y_idx % yidx_max, X_idx % xidx_max)
    V1 = get_elements_by_indices_3D(V_field, Z_idx, (Y_idx + 1) % yidx_max, X_idx % xidx_max)
    W0 = get_elements_by_indices_3D(W_field, Z_idx, Y_idx % yidx_max, X_idx % xidx_max)
    W1 = get_elements_by_indices_3D(W_field, Z_idx + 1, Y_idx % yidx_max, X_idx % xidx_max)

    U_interp = U0 + (X/dx - X_idx) * (U1 - U0)
    V_interp = V0 + (Y/dy - Y_idx) * (V1 - V0)
    W_interp = W0 + (Z - Zl[Z_idx])/dz * (W1 - W0)

    return U_interp, V_interp, W_interp


def rk4_static(ds, X, Y, dt):
    """
    Integrate the velocity field using the Runge-Kutta 4 method. Assumes a static velocity field. 

    Parameters
    ----------
    - ds: Dataset or any required argument for the UV_interp function, providing necessary data for interpolation.
    - X, Y: Current positions.
    - dt: Timestep for integration.

    Returns:
    - X_new, Y_new: New positions after a single RK4 step.
    """
    
    k1_U, k1_V = UV_interp(ds, X, Y)
    k2_U, k2_V = UV_interp(ds, X + 0.5 * dt * k1_U, Y + 0.5 * dt * k1_V)
    k3_U, k3_V = UV_interp(ds, X + 0.5 * dt * k2_U, Y + 0.5 * dt * k2_V)
    k4_U, k4_V = UV_interp(ds, X + dt * k3_U, Y + dt * k3_V)

    X_new = X + dt * (k1_U + 2*k2_U + 2*k3_U + k4_U) / 6
    Y_new = Y + dt * (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6

    return X_new, Y_new


@njit
def rk4_static_numpy(U_field, V_field, X, Y, dt, T:int):
    """
    Integrate the velocity field using the Runge-Kutta 4 method.

    Parameters
    ----------
    U_field : numpy array
        x-velocity field
    V_field : numpy array
        y-velocity field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point
    dt : int
        Timestep for integration.
    T : int
        Time in seconds; not used
    
    Returns
    -------
    X_new, Y_new : float; numpy array
        New positions after a single RK4 step.
    """
    k1_U, k1_V = UV_interp_numpy(U_field, V_field, X, Y)
    k2_U, k2_V = UV_interp_numpy(U_field, V_field, X + 0.5 * dt * k1_U, Y + 0.5 * dt * k1_V)
    k3_U, k3_V = UV_interp_numpy(U_field, V_field, X + 0.5 * dt * k2_U, Y + 0.5 * dt * k2_V)
    k4_U, k4_V = UV_interp_numpy(U_field, V_field, X + dt * k3_U, Y + dt * k3_V)

    X_new = X + dt * (k1_U + 2*k2_U + 2*k3_U + k4_U) / 6
    Y_new = Y + dt * (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6

    return X_new, Y_new

@njit
def rk4_3D_static_numpy(U_field, V_field, W_field, X, Y, Z, dt, T:int):
    """
    Integrate the velocity field using the Runge-Kutta 4 method.

    Parameters
    ----------
    U_field : numpy array
        x-velocity field
    V_field : numpy array
        y-velocity field
    W_field : numpy array
        z-velocity field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point
    Z : float; numpy array
        z-coordinate of the point
    dt : int
        Timestep for integration.
    T : int
        Time in seconds; not used
    
    Returns
    -------
    X_new, Y_new : float; numpy array
        New positions after a single RK4 step.
    """
    k1_U, k1_V, k1_W = UVW_interp_numpy(U_field, V_field, W_field, X, Y, Z)
    k2_U, k2_V, k2_W = UVW_interp_numpy(U_field, V_field, W_field, X + 0.5 * dt * k1_U, Y + 0.5 * dt * k1_V, Z + 0.5 * dt * k1_W)
    k3_U, k3_V, k3_W = UVW_interp_numpy(U_field, V_field, W_field, X + 0.5 * dt * k2_U, Y + 0.5 * dt * k2_V, Z + 0.5 * dt * k2_W)
    k4_U, k4_V, k4_W = UVW_interp_numpy(U_field, V_field, W_field, X + dt * k3_U, Y + dt * k3_V, Z + dt * k3_W)

    X_new = X + dt * (k1_U + 2*k2_U + 2*k3_U + k4_U) / 6
    Y_new = Y + dt * (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6
    Z_new = Z + dt * (k1_W + 2*k2_W + 2*k3_W + k4_W) / 6

    return X_new, Y_new, Z_new


@njit
def rk4_dynamic_numpy(U_field, V_field, X, Y, dt, T:int):
    """
    Integrate the velocity field using the Runge-Kutta 4 method.

    Parameters
    ----------
    U_field : numpy array
        x-velocity field
    V_field : numpy array
        y-velocity field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point
    dt : int
        Timestep for integration.
    T : int
        Time in seconds to interpolate to.
    
    Returns
    -------
    X_new, Y_new : float; numpy array
        New positions after a single RK4 step.
    """
    
    U_0, V_0 = time_interpolate_numpy(U_field, V_field, T)
    U_05, V_05 = time_interpolate_numpy(U_field, V_field, T + 0.5 * dt)
    U_1, V_1 = time_interpolate_numpy(U_field, V_field, T + dt)

    k1_U, k1_V = UV_interp_numpy(U_0, V_0, X, Y)
    k1 = (k1_U, k1_V)
    k2_U, k2_V = UV_interp_numpy(U_05, V_05, X + 0.5 * dt * k1_U, Y + 0.5 * dt * k1_V)
    k2 = (k2_U, k2_V)
    k3_U, k3_V = UV_interp_numpy(U_05, V_05, X + 0.5 * dt * k2_U, Y + 0.5 * dt * k2_V)
    k3 = (k3_U, k3_V)
    k4_U, k4_V = UV_interp_numpy(U_1, V_1, X + dt * k3_U, Y + dt * k3_V)
    k4 = (k4_U, k4_V)

    X_new = X + dt * (k1_U + 2*k2_U + 2*k3_U + k4_U) / 6
    Y_new = Y + dt * (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6

    return X_new, Y_new

@njit
def rk4_3D_dynamic_numpy(U_field, V_field, W_field, X, Y, Z, dt, T:int):
    """
    Integrate the velocity field using the Runge-Kutta 4 method.

    Parameters
    ----------
    U_field : numpy array
        x-velocity field
    V_field : numpy array
        y-velocity field
    W_field : numpy array
        z-velocity field
    X : float; numpy array
        x-coordinate of the point
    Y : float; numpy array
        y-coordinate of the point
    Z : float; numpy array
        z-coordinate of the point
    dt : int
        Timestep for integration.
    T : int
        Time in seconds; not used
    
    Returns
    -------
    X_new, Y_new : float; numpy array
        New positions after a single RK4 step.
    """
    U_0, V_0, W_0 = time_interpolate_numpy_3D(U_field, V_field, W_field, T)
    U_05, V_05, W_05 = time_interpolate_numpy_3D(U_field, V_field, W_field, T + 0.5 * dt)
    U_1, V_1, W_1 = time_interpolate_numpy_3D(U_field, V_field, W_field, T + dt)

    k1_U, k1_V, k1_W = UVW_interp_numpy(U_0, V_0, W_0, X, Y, Z)
    k2_U, k2_V, k2_W = UVW_interp_numpy(U_05, V_05, W_05, X + 0.5 * dt * k1_U, Y + 0.5 * dt * k1_V, Z + 0.5 * dt * k1_W)
    k3_U, k3_V, k3_W = UVW_interp_numpy(U_05, V_05, W_05, X + 0.5 * dt * k2_U, Y + 0.5 * dt * k2_V, Z + 0.5 * dt * k2_W)
    k4_U, k4_V, k4_W = UVW_interp_numpy(U_1, V_1, W_1, X + dt * k3_U, Y + dt * k3_V, Z + dt * k3_W)

    X_new = X + dt * (k1_U + 2*k2_U + 2*k3_U + k4_U) / 6
    Y_new = Y + dt * (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6
    Z_new = Z + dt * (k1_W + 2*k2_W + 2*k3_W + k4_W) / 6

    return X_new, Y_new, Z_new


def integration_loop_2D_numpy(integrator, 
                           pset = None,
                           X0 = None, 
                           Y0 = None, 
                           U_field = None, 
                           V_field = None, 
                           dt=60, 
                           T_int=24*60*60, 
                           T0=0, 
                           output_dt=24*60*60, 
                           data_frequency=24*60*60):
    """
    Integrate the velocity field.

    Parameters
    ----------
    integrator : function
        The integration function to use
    X0 : numpy array
        x-coordinates of the initial positions
    Y0 : numpy array
        y-coordinates of the initial positions
    U_field : numpy array
        x-velocity field
    V_field : numpy array
        y-velocity field
    dt : int
        Timestep for integration
    T_int : int
        Total time to integrate for
    T0 : int
        Initial time in seconds
    output_dt : int
        Output frequency
    data_frequency : int
        Frequency of the velocity field data

    Returns
    -------
    X : np.array
        Array of x-locations
    Y : np.array
        Array of y-locations
    T : np.array
        Array of times
    """
    if pset is None:
        assert X0 is not None and Y0 is not None, "Initial positions must be supplied"
    else:
        X0 = pset.x_init
        Y0 = pset.y_init

    assert T_int % output_dt == 0, "T must be divisible by output_dt"
    assert T_int % dt == 0, "T must be divisible by dt"
    assert T_int % output_dt == 0, "T_int must be divisible by output_dt"
    
    n_T = round(T_int / np.abs(dt))
    n_T_out = round(T_int / output_dt) + 1

    n_parts = len(X0)

    X = np.zeros((n_T_out, n_parts)) * np.nan
    Y = np.zeros((n_T_out, n_parts)) * np.nan
    T_out = np.zeros(n_T_out)

    X[0] = X0.astype(np.float64)
    Y[0] = Y0.astype(np.float64)
    T_out[0] = T0

    T = T0
    idx_out = 0

    X_step = X0.astype(np.float64).copy()
    Y_step = Y0.astype(np.float64).copy()

    for i in tqdm.tqdm(range(1, n_T + 1), desc="Integrating"):
        X_step, Y_step = integrator(U_field, V_field, X_step, Y_step, dt, T)
        T += dt
        if T % output_dt == 0:
            idx_out += 1
            X[idx_out, :] = X_step
            Y[idx_out, :] = Y_step
            T_out[idx_out] = T

    if pset is None:
        return X, Y, T_out
    else:
        pset.store_results(X=X, Y=Y, T=T_out)
        pset.add_metadata("function", "integration_loop_2D_numpy")
        pset.add_metadata("integrator", integrator.__name__)
        pset.add_metadata("dt", dt)
        pset.add_metadata("T_int", T_int)
        pset.add_metadata("T0", T0) 
        pset.add_metadata("output_dt", output_dt)


def integration_loop_3D_numpy(integrator, 
                              pset = None,
                              X0 = None, 
                              Y0 = None, 
                              Z0 = None, 
                              U_field = None, 
                              V_field = None,
                              W_field = None, 
                              dt=60, 
                              T_int=24*60*60, 
                              T0=0, 
                              output_dt=24*60*60, 
                              data_frequency=24*60*60):
    """
    Integrate the velocity field.

    Parameters
    ----------
    integrator : function
        The integration function to use
    X0 : numpy array
        x-coordinates of the initial positions
    Y0 : numpy array
        y-coordinates of the initial positions
    Z0 : numpy array
        z-coordinates of the initial positions
    U_field : numpy array
        x-velocity field
    V_field : numpy array
        y-velocity field
    W_field : numpy array
        z-velocity field
    dt : int
        Timestep for integration
    T_int : int
        Total time to integrate for
    T0 : int
        Initial time in seconds
    output_dt : int
        Output frequency
    data_frequency : int
        Frequency of the velocity field data

    Returns
    -------
    X : np.array
        Array of x-locations
    Y : np.array
        Array of y-locations
    Z : np.array
        Array of y-locations
    T : np.array
        Array of times
    """
    if pset is None:
        assert X0 is not None and Y0 is not None and Z0 is not None, "Initial positions must be supplied"
    else:
        X0 = pset.x_init
        Y0 = pset.y_init
        Z0 = pset.z_init
    assert T_int % output_dt == 0, "T must be divisible by output_dt"
    assert T_int % dt == 0, "T must be divisible by dt"
    assert T_int % output_dt == 0, "T_int must be divisible by output_dt"
    
    n_T = round(T_int / np.abs(dt))
    n_T_out = round(T_int / output_dt) + 1

    n_parts = len(X0)

    X = np.zeros((n_T_out, n_parts)) * np.nan
    Y = np.zeros((n_T_out, n_parts)) * np.nan
    Z = np.zeros((n_T_out, n_parts)) * np.nan
    T_out = np.zeros(n_T_out)

    X[0] = X0.astype(np.float64)
    Y[0] = Y0.astype(np.float64)
    Z[0] = Z0.astype(np.float64)
    T_out[0] = T0

    T = T0
    idx_out = 0

    X_step = X0.astype(np.float64).copy()
    Y_step = Y0.astype(np.float64).copy()
    Z_step = Z0.astype(np.float64).copy()

    for i in tqdm.tqdm(range(1, n_T + 1), desc="Integrating"):
        X_step, Y_step, Z_step = integrator(U_field, V_field, W_field, X_step, Y_step, Z_step, dt, T)
        Z_step = np.clip(Z_step, -5000, 0) # Prevent through surface
        T += dt
        if T % output_dt == 0:
            idx_out += 1
            X[idx_out, :] = X_step
            Y[idx_out, :] = Y_step
            Z[idx_out, :] = Z_step
            T_out[idx_out] = T

    if pset is None:
        return X, Y, Z, T_out
    else:
        pset.store_results(X=X, Y=Y, Z=Z, T=T_out)
        pset.add_metadata("function", "integration_loop_3D_numpy")
        pset.add_metadata("integrator", integrator.__name__)
        pset.add_metadata("dt", dt)
        pset.add_metadata("T_int", T_int)
        pset.add_metadata("T0", T0) 
        pset.add_metadata("output_dt", output_dt)

"""
Below is code for the analytical scheme as described in Döös et al. (2017). 
The time-varying case is implemented similar to in ARIANE, where the velocity
field is subdivided into interpolated slices, and the scheme is piecewise continuous.
"""

@njit
def find_UV_boundaries(x, y, U_field, V_field):
    """
    Find U and V at the boundaries, given a location. 
    For points at the boundary, takes into account which cell to 
    consider (on the left or right), based on where the particle should move to.

    Parameters
    ----------
    x : float
        x-location
    y : float
        y-location
    U_field : np.array (ny, nx)
        Stationary U-data
    V_field : np.array (ny, nx)
        Stationary V-data

    Returns
    -------
    U0 : float
        Velocity at western wall
    U1 : float
        Velocity at eastern wall
    V0 : float
        Velocity at southern wall
    V1 : float
        Velocity at northern wall
    XG_idx : int
        idx of western wall
    YG_idx : int
        idx of southern wall
    """

    XG_idx = int(math.floor(x / dx))
    YG_idx = int(math.floor(y / dy))

    U0 = U_field[YG_idx % yidx_max, XG_idx % xidx_max]
    V0 = V_field[YG_idx % yidx_max, XG_idx % xidx_max]

    # If at the cell boundary, check if the velocity is negative. If so, decrease the index by 1 (boundary is 'right' neighbor)
    decrease_XG = 0
    decrease_YG = 0
    if x % dx == 0:
        if U0 > 0:
            pass
        else:
            decrease_XG = 1
    if y % dy == 0:
        if V0 > 0:
            pass
        else:
            decrease_YG = 1
    
    if decrease_XG or decrease_YG:
        XG_idx -= decrease_XG
        YG_idx -= decrease_YG
        U0 = U_field[YG_idx % yidx_max, XG_idx % xidx_max]
        V0 = V_field[YG_idx % yidx_max, XG_idx % xidx_max]

    U1 = U_field[YG_idx % yidx_max, (XG_idx + 1) % xidx_max]
    V1 = V_field[(YG_idx + 1) % yidx_max, XG_idx % xidx_max]

    return U0, U1, V0, V1, XG_idx, YG_idx


@njit
def find_UV_boundaries_3D(x, y, z, U_field, V_field, W_field):
    """
    Find U and V at the boundaries, given a location. 
    For points at the boundary, takes into account which cell to 
    consider (on the left or right), based on where the particle should move to.

    TODO:
     - This function can be optimized by taking the previous 'timestep' into
       account (except for the initialization timestep).

    Parameters
    ----------
    x : float
        x-location
    y : float
        y-location
    z : float
        z-location
    U_field : np.array (nz, ny, nx)
        Stationary U-data
    V_field : np.array (nz, ny, nx)
        Stationary V-data
    W_field : np.array (nz, ny, nx)

    Returns
    -------
    U0 : float
        Velocity at western wall
    U1 : float
        Velocity at eastern wall
    V0 : float
        Velocity at southern wall
    V1 : float
        Velocity at northern wall
    W0 : float
        Velocity at upper wall
    W1 : float
        Velocity at lower wall
    XG_idx : int
        idx of western wall
    YG_idx : int
        idx of southern wall
    ZG_idx : int
        idx of upper wall
    """

    XG_idx = int(np.floor(x / dx))
    YG_idx = int(np.floor(y / dy))
    ZG_idx = np.searchsorted(-Zl, -z, side='right') - 1     # Hard-coded for MITgcm


    U0 = U_field[ZG_idx, YG_idx % yidx_max, XG_idx % xidx_max]
    V0 = V_field[ZG_idx, YG_idx % yidx_max, XG_idx % xidx_max]
    W0 = W_field[ZG_idx, YG_idx % yidx_max, XG_idx % xidx_max]

    # If at the cell boundary, check if the velocity is negative. If so, decrease the index by 1 (boundary is 'right' neighbor)
    decrease_XG = 0
    decrease_YG = 0
    decrease_ZG = 0
    if x % dx == 0:
        if U0 > 0:
            pass
        else:
            decrease_XG = 1
    if y % dy == 0:
        if V0 > 0:
            pass
        else:
            decrease_YG = 1
    if z == Zl[ZG_idx]:
        if W0 < 0:  # Note the direction
            pass
        else:
            decrease_ZG = 1

    
    if decrease_XG or decrease_YG or decrease_ZG:
        XG_idx -= decrease_XG
        YG_idx -= decrease_YG
        ZG_idx -= decrease_ZG
        U0 = U_field[ZG_idx, YG_idx % yidx_max, XG_idx % xidx_max]
        V0 = V_field[ZG_idx, YG_idx % yidx_max, XG_idx % xidx_max]
        W0 = W_field[ZG_idx, YG_idx % yidx_max, XG_idx % xidx_max]

    U1 = U_field[ZG_idx, YG_idx % yidx_max, (XG_idx + 1) % xidx_max]
    V1 = V_field[ZG_idx, (YG_idx + 1) % yidx_max, XG_idx % xidx_max]
    W1 = W_field[ZG_idx + 1, YG_idx % yidx_max, XG_idx % xidx_max]

    return U0, U1, V0, V1, W0, W1, XG_idx, YG_idx, ZG_idx


@njit
def find_next_position(x_t0, y_t0, U_field, V_field, dt_max=np.inf):
    """
    Given an initial position and velocity field, compute the next position
    using the analytical method. Based on Döös et al. (2017).

    Parameters
    ----------
    x_t0 : float
        initial x-position
    y_t0 : float
        initial y-position
    U_field : np.array (ny, nx)
        U-velocity
    V_field : np.array (ny, nx)
        V-velocity
    dt_max : float
        Maximum integration time. If the found crossing time is higher than
        dt_max, use dt_max to find the next position

    Returns
    -------
    x_t1 : float
        new x-position
    y_t1, : float
        new y-position
    dt : float
        Time taken between positions
    """
    
    U0, U1, V0, V1, XG_idx, YG_idx = find_UV_boundaries(x_t0, y_t0, U_field, V_field)

    alpha_x = -(U1 - U0)/dx
    beta_x = (XG_idx) * (U1 - U0) - U0  # Note that XG_idx is X_left_cell/dx
    
    # X-direction
    if x_t0 % dx == 0:                  # Coming through the U-wall
        if x_t0 // dx == XG_idx:        # Coming from the left
            x_t1 = (XG_idx + 1) * dx
        elif x_t0 // dx == XG_idx + 1:  # Coming from the right
            x_t1 = (XG_idx) * dx
        else:
            raise ValueError("Invalid x_t0")
    else:                               # Coming through the V-wall
        x_t1 = (int(np.sign(U0 + (x_t0%dx)/dx * (U1 - U0))/2 + 0.5) + XG_idx) * dx
    
    if alpha_x == 0:
        t1_x = (x_t0 - x_t1) / beta_x
    else:
        Fx1 = (x_t1 + beta_x/alpha_x)
        Fx0 = (x_t0 + beta_x/alpha_x)
        if np.sign(Fx1) != np.sign(Fx0):
            # Zero transport somewhere in the box
            t1_x = np.inf
        else:
            t1_x = - 1/alpha_x * np.log(Fx1/Fx0)
            if t1_x < 0:
                t1_x = np.inf

    # Y-direction
    alpha_y = -(V1 - V0)/dy
    beta_y = (YG_idx) * (V1 - V0) - V0  # Note that YG_idx is Y_left_cell/dy
    if y_t0 % dy == 0:                  # Coming through the V-wall
        if y_t0 // dy == YG_idx:        # coming from the bottom
            y_t1 = (YG_idx + 1) * dy
        elif y_t0 // dy == YG_idx + 1:  # coming from the top
            y_t1 = (YG_idx) * dy
        else:
            raise ValueError("Invalid y_t0")
    else:                               # Coming through U-wall
        y_t1 = (int(np.sign(V0 + (y_t0%dy)/dy * (V1 - V0))/2 + 0.5) + YG_idx) * dy
    

    if alpha_y == 0:
        t1_y = (y_t0 - y_t1) / beta_y
    else:
        Fy1 = (y_t1 + beta_y/alpha_y)
        Fy0 = (y_t0 + beta_y/alpha_y)

        if np.sign(Fy1) != np.sign(Fy0):
            # Zero transport somewhere in the box
            t1_y = np.inf
        else:
            t1_y = - 1/alpha_y * np.log(Fy1/Fy0)
            if t1_y < 0:
                t1_y = np.inf

    # Computing transit time
    dt = min([t1_x, t1_y, np.float64(dt_max)])       
    # The following makes sure we stay exactly on one of the boundaries if any of the dt < dt_max
    if t1_x < t1_y and t1_x < dt_max:
        if alpha_y == 0:
            y_t1 = - beta_y * dt + y_t0
        else:
            y_t1 = (y_t0 + beta_y/alpha_y) * np.exp(-alpha_y * dt) - beta_y/alpha_y
    elif t1_y < t1_x and t1_y < dt_max:
        if alpha_x == 0:
            x_t1 = - beta_x * dt + x_t0
        else:
            x_t1 = (x_t0 + beta_x/alpha_x) * np.exp(-alpha_x * dt) - beta_x/alpha_x
    else:
        if alpha_x == 0:
            x_t1 = - beta_x * dt + x_t0
        else:
            x_t1 = (x_t0 + beta_x/alpha_x) * np.exp(-alpha_x * dt) - beta_x/alpha_x
        if alpha_y == 0:
            y_t1 = - beta_y * dt + y_t0
        else:
            y_t1 = (y_t0 + beta_y/alpha_y) * np.exp(-alpha_y * dt) - beta_y/alpha_y

    return x_t1, y_t1, dt


@njit
def find_next_position_3D(x_t0, y_t0, z_t0, U_field, V_field, W_field, dt_max=np.inf):
    """
    Given an initial position and velocity field, compute the next position
    using the analytical method. Based on Döös et al. (2017).

    Parameters
    ----------
    x_t0 : float
        initial x-position
    y_t0 : float
        initial y-position
    z_t0 : float
        initial z-position
    U_field : np.array (nz, ny, nx)
        U-velocity
    V_field : np.array (nz, ny, nx)
        V-velocity
    W_field : np.array (nz, ny, nx)
        W-velocity
    dt_max : float
        Maximum integration time. If the found crossing time is higher than
        dt_max, use dt_max to find the next position

    Returns
    -------
    x_t1 : float
        new x-position
    y_t1, : float
        new y-position
    z_t1 : float
        new z-position
    dt : float
        Time taken between positions
    """
    
    U0, U1, V0, V1, W0, W1, XG_idx, YG_idx, ZG_idx = find_UV_boundaries_3D(x_t0, y_t0, z_t0, U_field, V_field, W_field)

    # x-direction
    alpha_x = -(U1 - U0)/dx
    beta_x = (XG_idx) * (U1 - U0) - U0  # Note that XG_idx is X_left_cell/dx
    
    if x_t0 % dx == 0:                  # Coming through the U-wall
        if x_t0 // dx == XG_idx:        # Coming from the left
            x_t1 = (XG_idx + 1) * dx
        elif x_t0 // dx == XG_idx + 1:  # Coming from the right
            x_t1 = (XG_idx) * dx
        else:
            raise ValueError("Invalid x_t0")
    else:                               # Coming through the V-wall
        x_t1 = (int(np.sign(U0 + (x_t0%dx)/dx * (U1 - U0))/2 + 0.5) + XG_idx) * dx
    
    if alpha_x == 0:
        if beta_x == 0:
            t1_x = np.inf
        else:
            t1_x = (x_t0 - x_t1) / beta_x
    else:
        Fx1 = (x_t1 + beta_x/alpha_x)
        Fx0 = (x_t0 + beta_x/alpha_x)
        if np.sign(Fx1) != np.sign(Fx0) or Fx0 == 0 or Fx1 == 0:
            # Zero transport somewhere in the box
            t1_x = np.inf
        else:
            t1_x = - 1/alpha_x * np.log(Fx1/Fx0)
            if t1_x < 0:
                t1_x = np.inf

    # y-direction
    alpha_y = -(V1 - V0)/dy
    beta_y = (YG_idx) * (V1 - V0) - V0  # Note that YG_idx is Y_left_cell/dy
    if y_t0 % dy == 0:                  # Coming through the V-wall
        if y_t0 // dy == YG_idx:        # coming from the bottom
            y_t1 = (YG_idx + 1) * dy
        elif y_t0 // dy == YG_idx + 1:  # coming from the top
            y_t1 = (YG_idx) * dy
        else:
            raise ValueError("Invalid y_t0")
    else:                               # Coming through U-wall
        y_t1 = (int(np.sign(V0 + (y_t0%dy)/dy * (V1 - V0))/2 + 0.5) + YG_idx) * dy
    
    if alpha_y == 0:
        if beta_y == 0:
            t1_y = np.inf
        else:
            t1_y = (y_t0 - y_t1) / beta_y
    else:
        Fy1 = (y_t1 + beta_y/alpha_y)
        Fy0 = (y_t0 + beta_y/alpha_y)

        if np.sign(Fy1) != np.sign(Fy0) or Fy0 == 0 or Fy1 == 0:
            # Zero transport somewhere in the box
            t1_y = np.inf
        else:
            t1_y = - 1/alpha_y * np.log(Fy1/Fy0)
            if t1_y < 0:
                t1_y = np.inf

    # Z-direction
    dz = dz_arr[ZG_idx]

    alpha_z = -(W1 - W0)/dz
    beta_z = Zl[ZG_idx] * (W1 - W0) / dz - W0   # Differs from above method, since now dz is variable

    if z_t0 == Zl[ZG_idx]:                      # Coming from the top
        z_t1 = Zl[ZG_idx + 1]
    elif z_t0 == Zl[ZG_idx + 1]:                # Coming from the bottom
        z_t1 = Zl[ZG_idx]
    else:                                       # Coming through the V-wall
        z_t1 = Zl[ZG_idx + int(-np.sign(W0 + (z_t0 - Zl[ZG_idx]) / dz * (W1 - W0)) / 2 + 0.5)] # Note: negative W is pointed in positive z-direction.
    
    if alpha_z == 0:
        if beta_z == 0:
            t1_z = np.inf
        else:
            t1_z = (z_t0 - z_t1) / beta_z
    else:
        Fz1 = (z_t1 + beta_z/alpha_z)
        Fz0 = (z_t0 + beta_z/alpha_z)
        if np.sign(Fz1) != np.sign(Fz0) or Fz0 == 0 or Fz1 == 0:
            # Zero transport somewhere in the box
            t1_z = np.inf
        else:
            t1_z = - 1/alpha_z * np.log(Fz1/Fz0)
            if t1_z < 0:
                t1_z = np.inf

    # Computing transit time
    dt = min([t1_x, t1_y, t1_z, np.float64(dt_max)])

    # np.argmin doesn't seem to work for some reason, so:
       
    # The following makes sure we stay exactly on one of the boundaries if any of the dt < dt_max
    if t1_x < t1_y and t1_x < t1_z and t1_x < dt_max:
        if alpha_y == 0:
            y_t1 = - beta_y * dt + y_t0
        else:
            y_t1 = (y_t0 + beta_y/alpha_y) * np.exp(-alpha_y * dt) - beta_y/alpha_y
        if alpha_z == 0:
            z_t1 = - beta_z * dt + z_t0
        else:
            z_t1 = (z_t0 + beta_z/alpha_z) * np.exp(-alpha_z * dt) - beta_z/alpha_z
    elif t1_y < t1_x and t1_y < t1_z and t1_y < dt_max:
        if alpha_x == 0:
            x_t1 = - beta_x * dt + x_t0
        else:
            x_t1 = (x_t0 + beta_x/alpha_x) * np.exp(-alpha_x * dt) - beta_x/alpha_x
        if alpha_z == 0:
            z_t1 = - beta_z * dt + z_t0
        else:
            z_t1 = (z_t0 + beta_z/alpha_z) * np.exp(-alpha_z * dt) - beta_z/alpha_z
    elif t1_z < t1_x and t1_z < t1_y and t1_z < dt_max:
        if alpha_x == 0:
            x_t1 = - beta_x * dt + x_t0
        else:
            x_t1 = (x_t0 + beta_x/alpha_x) * np.exp(-alpha_x * dt) - beta_x/alpha_x
        if alpha_y == 0:
            y_t1 = - beta_y * dt + y_t0
        else:
            y_t1 = (y_t0 + beta_y/alpha_y) * np.exp(-alpha_y * dt) - beta_y/alpha_y
    else:
        if alpha_x == 0:
            x_t1 = - beta_x * dt + x_t0
        else:
            x_t1 = (x_t0 + beta_x/alpha_x) * np.exp(-alpha_x * dt) - beta_x/alpha_x
        if alpha_y == 0:
            y_t1 = - beta_y * dt + y_t0
        else:
            y_t1 = (y_t0 + beta_y/alpha_y) * np.exp(-alpha_y * dt) - beta_y/alpha_y
        if alpha_z == 0:
            z_t1 = - beta_z * dt + z_t0
        else:
            z_t1 = (z_t0 + beta_z/alpha_z) * np.exp(-alpha_z * dt) - beta_z/alpha_z

    # Prevent going through the surface
    if z_t1 > 0:
        z_t1 = 0

    return x_t1, y_t1, z_t1, dt


def integration_loop_analytical_2D_static(pset=None, 
                                        X0=None, 
                                        Y0=None, 
                                        U_field=None, 
                                        V_field=None, 
                                        T_int=24*60*60, 
                                        output_dt=24*60*60, 
                                        maxiter=10000):
    """
    Integrate the velocity field using the analytical method. 
    This function only works for the stationary case.

    TODO
     - add initial time option

    Parameters
    ----------
    pset : Lm.pset
        Particle set to load locations from and store result to.
    X0 : np.array
        Initial x-positions
    Y0 : np.array
        Initial y-positions
    U_field : np.array
        U-velocity fields
    V_field : np.array
        V-velocity fields
    T_int : int
        total integration time
    output_dt : int
        output frequency
    maxiter : int
        Maximum number of integrations per particle (prevent stationary orbits).

    Returns
    -------
    X : np.array
        Array of x-locations
    Y : np.array
        Array of y-locations
    T : np.array
        Array of times
    """
    assert T_int % output_dt == 0, "T must be divisible by output_dt"
    assert T_int % output_dt == 0, "T_int must be divisible by output_dt"

    if pset is None:
        assert X0 is not None and Y0 is not None, "Initial positions must be supplied"
    else:
        X0 = pset.x_init
        Y0 = pset.y_init
    n_T_out = round(T_int / output_dt) + 1

    n_parts = len(X0)

    X = np.zeros((n_T_out, n_parts), dtype=np.float64) * np.nan
    Y = np.zeros((n_T_out, n_parts), dtype=np.float64) * np.nan
    T_out = np.arange(0, T_int + output_dt, output_dt)

    X[0] = X0
    Y[0] = Y0

    for partidx in tqdm.tqdm(range(n_parts), desc="Integrating over particles"):
        tindex = 1
        timer = 0
        iteration = 0

        X_step = X0[partidx]
        Y_step = Y0[partidx]

        if np.isnan(X_step) or np.isnan(Y_step):
            continue

        while timer < T_int and iteration < maxiter:
            X_step, Y_step, dt = find_next_position(X_step, Y_step, U_field, V_field, dt_max=T_out[tindex] - timer)
            timer += dt
            iteration += 1
            if T_out[tindex] - timer < 1e-5:
                X[tindex, partidx] = X_step
                Y[tindex, partidx] = Y_step
                tindex +=1

    if pset is None:
        return X, Y, T_out
    else:
        pset.store_results(X=X, Y=Y, T=T_out)
        pset.add_metadata("function", "integration_loop_analytical_2D_static")
        pset.add_metadata("maxiter", maxiter)
        pset.add_metadata("T_int", T_int)
        # pset.add_metadata("T0", T0) 
        pset.add_metadata("output_dt", output_dt)


def integration_loop_analytical_3D_static(pset=None, 
                                          X0=None, 
                                          Y0=None, 
                                          Z0=None, 
                                          U_field=None, 
                                          V_field=None, 
                                          W_field=None, 
                                          T_int=24*60*60, 
                                          output_dt=24*60*60, 
                                          maxiter=10000):
    """
    Integrate the velocity field using the analytical method. 
    This function only works for the stationary case.

    TODO
     - add initial time option

    Parameters
    ----------
    X0 : np.array
        Initial x-positions
    Y0 : np.array
        Initial y-positions
    Z0 : np.array
        Initial z-positions
    U_field : np.array
        U-velocity fields
    V_field : np.array
        V-velocity fields
    W_field : np.array
        W-velocity fields
    T_int : int
        total integration time
    output_dt : int
        output frequency
    maxiter : int
        Maximum number of integrations per particle (prevent stationary orbits).

    Returns
    -------
    X : np.array
        Array of x-locations
    Y : np.array
        Array of y-locations
    Z : np.array
        Array of z-location
    T : np.array
        Array of times
    """
    if pset is None:
        assert X0 is not None and Y0 is not None and Z0 is not None, "Initial positions must be supplied"
    else:
        X0 = pset.x_init
        Y0 = pset.y_init
        Z0 = pset.z_init
    assert T_int % output_dt == 0, "T must be divisible by output_dt"
    assert T_int % output_dt == 0, "T_int must be divisible by output_dt"
    
    n_T_out = round(T_int / output_dt) + 1

    n_parts = len(X0)

    X = np.zeros((n_T_out, n_parts), dtype=np.float64) * np.nan
    Y = np.zeros((n_T_out, n_parts), dtype=np.float64) * np.nan
    Z = np.zeros((n_T_out, n_parts), dtype=np.float64) * np.nan
    T_out = np.arange(0, T_int + output_dt, output_dt)

    X[0] = X0
    Y[0] = Y0
    Z[0] = Z0

    for partidx in tqdm.tqdm(range(n_parts), desc="Integrating over particles"):
        tindex = 1
        timer = 0
        iteration = 0

        X_step = X0[partidx]
        Y_step = Y0[partidx]
        Z_step = Z0[partidx]

        if np.isnan(X_step) or np.isnan(Y_step) or np.isnan(Z_step):
            continue

        while timer < T_int and iteration < maxiter:
            X_step, Y_step, Z_step, dt = find_next_position_3D(X_step, Y_step, Z_step, U_field, V_field, W_field, dt_max=T_out[tindex] - timer)
            timer += dt
            iteration += 1
            if T_out[tindex] - timer < 1e-5:
                X[tindex, partidx] = X_step
                Y[tindex, partidx] = Y_step
                Z[tindex, partidx] = Z_step
                tindex +=1

    if pset is None:
        return X, Y, Z, T_out
    else:
        pset.store_results(X=X, Y=Y, Z=Z, T=T_out)
        pset.add_metadata("function", "integration_loop_analytical_3D_static")
        pset.add_metadata("maxiter", maxiter)
        pset.add_metadata("T_int", T_int)
        # pset.add_metadata("T0", T0) 
        pset.add_metadata("output_dt", output_dt)


def integration_loop_analytical_2D_dynamic(pset = None,
                                X0 = None, 
                                Y0 = None, 
                                U_field = None, 
                                V_field = None, 
                                dt = 5 * 60, 
                                T_int=24*60*60, 
                                T0=0, 
                                output_dt=24*60*60,
                                data_frequency=24*60*60, 
                                maxiter=10000,
                                method = 'interp_slab'):
    """
    Integrate the velocity field using the analytical method

    Parameters
    ----------
    pset : Lm.pset
        Particleset to initalize with and store results in.
    X0 : numpy array
        x-coordinates of the initial positions
    Y0 : numpy array
        y-coordinates of the initial positions
    U_field : numpy array
        x-velocity field
    V_field : numpy array
        y-velocity field
    dt : int
        Timestep for interpolating 'slabs' of constant U
    T_int : int
        Total time to integrate for
    T0 : int
        Initial time in seconds
    output_dt : int
        Output frequency
    data_frequency : int
        Frequency of the velocity field data
    maxiter : int
        Maximum number of iterations (useful in case of stationary orbits)
    method : str
        Either 'interp_slab' or 'ARIANE'. In the case of 'interp_slab', model output is interpolated to stationary 'time slabs'. 
        In case of 'ARIANE', just take the model output and don't interpolate.
    """
    if pset is None:
        assert X0 is not None and Y0 is not None, "Initial positions must be supplied"
    else:
        X0 = pset.x_init
        Y0 = pset.y_init
    assert T_int % output_dt == 0, "T must be divisible by output_dt"
    assert T_int % dt == 0, "T must be divisible by dt"
    assert T_int % output_dt == 0, "T_int must be divisible by output_dt"
    assert data_frequency % dt == 0, "data_frequency must be divisible by dt"

    n_T = round(T_int / np.abs(dt))
    n_T_out = round(T_int / output_dt) + 1

    n_parts = len(X0)

    X = np.zeros((n_T_out, n_parts), dtype=np.float64) * np.nan
    Y = np.zeros((n_T_out, n_parts), dtype=np.float64) * np.nan
    T_out = np.zeros(n_T_out, dtype=np.float64)

    X[0] = X0
    Y[0] = Y0
    T_out[0] = np.float64(T0)

    T = np.float64(T0)
    idx_out = 0

    X_step = X0.astype(np.float64).copy()
    Y_step = Y0.astype(np.float64).copy()

    if len(U_field.shape) == 2:
        interp = False
        U_field_step = U_field
        V_field_step = V_field
        if dt != output_dt:
            print("It's best to make dt = output_dt for numerical accuracy")
    elif len(U_field.shape) == 3:
        interp = True
    else:
        raise ValueError("Velocity field must have 3 or 2 dimensions.")
        

    if dt < 0:
        U_field = - U_field
        V_field = - V_field
    

    for i in tqdm.tqdm(range(1, n_T + 1), desc="Integrating timesteps"):
        if interp:
            if method == "interp_slab":
                U_field_step, V_field_step = time_interpolate_numpy(U_field, V_field, T + dt / 2, data_frequency)
            elif method == "ARIANE":
                U_field_step, V_field_step = time_select_numpy(U_field, V_field, T + dt / 2, data_frequency, dt=dt)
            else:
                raise ValueError("Mode must be either `interp_slab` or `ARIANE`")
        for partidx in range(n_parts):
            timer_sub = 0
            subiter = 0
            while timer_sub < abs(dt) and subiter < maxiter:
                if np.isnan(X_step[partidx]) or np.isnan(Y_step[partidx]):
                    break
                X_step[partidx], Y_step[partidx], dt_sub = find_next_position(X_step[partidx], Y_step[partidx], U_field_step, V_field_step, dt_max=abs(dt) - timer_sub)
                timer_sub += dt_sub
                subiter += 1
                if subiter == maxiter:
                    print(f"Particle {partidx} reached max iter")
                    X_step[partidx] = np.nan
                    Y_step[partidx] = np.nan
        T += dt
        if T % output_dt == 0:
            idx_out += 1
            X[idx_out, :] = X_step
            Y[idx_out, :] = Y_step
            T_out[idx_out] = T

    if pset is None:
        return X, Y, T_out
    else:
        pset.store_results(X=X, Y=Y, T=T_out)
        pset.add_metadata("function", "integration_loop_analytical_2D_dynamic")
        pset.add_metadata("dt", dt)
        pset.add_metadata("maxiter", maxiter)
        pset.add_metadata("T_int", T_int)
        pset.add_metadata("T0", T0) 
        pset.add_metadata("output_dt", output_dt)
        

def integration_loop_analytical_3D_dynamic(pset = None,
                                    X0 = None, 
                                    Y0 = None, 
                                    Z0 = None, 
                                    U_field = None, 
                                    V_field = None, 
                                    W_field = None, 
                                    dt = 5 * 60, 
                                    T_int=24*60*60, 
                                    T0=0, 
                                    output_dt=24*60*60, 
                                    data_frequency=24*60*60, 
                                    maxiter=10000):
    """
    Integrate the velocity field using the analytical method

    Parameters
    ----------
    pset : Lm.pset
        Particleset to initalize from and store results to.
    X0 : numpy array
        x-coordinates of the initial positions
    Y0 : numpy array
        y-coordinates of the initial positions
    Z0 : numpy array
        z-coordinates of the initial positions
    U_field : numpy array
        x-velocity field
    V_field : numpy array
        y-velocity field
    W_field : numpy array
        z-velocity field
    dt : int
        Timestep for interpolating 'slabs' of constant U
    T_int : int
        Total time to integrate for
    T0 : int
        Initial time in seconds
    output_dt : int
        Output frequency
    data_frequency : int
        Frequency of the velocity field data
    maxiter : int
        Maximum number of iterations (useful in case of stationary orbits)
    """
    if pset is None:
        assert X0 is not None and Y0 is not None and Z0 is not None, "Initial positions must be supplied"
    else:
        X0 = pset.x_init
        Y0 = pset.y_init
        Z0 = pset.z_init
    assert T_int % output_dt == 0, "T must be divisible by output_dt"
    assert T_int % dt == 0, "T must be divisible by dt"
    assert T_int % output_dt == 0, "T_int must be divisible by output_dt"
    assert data_frequency % dt == 0, "data_frequency must be divisible by dt"

    n_T = round(T_int / np.abs(dt))
    n_T_out = round(T_int / output_dt) + 1

    n_parts = len(X0)

    X = np.zeros((n_T_out, n_parts), dtype=np.float64) * np.nan
    Y = np.zeros((n_T_out, n_parts), dtype=np.float64) * np.nan
    Z = np.zeros((n_T_out, n_parts), dtype=np.float64) * np.nan
    T_out = np.zeros(n_T_out, dtype=np.float64)

    X[0] = X0
    Y[0] = Y0
    Z[0] = Z0
    T_out[0] = np.float64(T0)

    T = np.float64(T0)
    idx_out = 0

    X_step = X0.astype(np.float64).copy()
    Y_step = Y0.astype(np.float64).copy()
    Z_step = Z0.astype(np.float64).copy()

    if len(U_field.shape) == 3:
        interp = False
        U_field_step = U_field
        V_field_step = V_field
        W_field_step = W_field
        if dt != output_dt:
            print("It's best to make dt = output_dt for numerical accuracy")
    elif len(U_field.shape) == 4:
        interp = True
    else:
        raise ValueError("Velocity field must have 4 (time, nz, ny, nx) or 3 (time, ny, nx) dimensions.")
        

    if dt < 0:
        U_field = - U_field
        V_field = - V_field
        W_field = - W_field
    

    for i in tqdm.tqdm(range(1, n_T + 1), desc="Integrating timesteps"):
        if interp:
            U_field_step, V_field_step, W_field_step = time_interpolate_numpy_3D(U_field, V_field, W_field, T + dt / 2, data_frequency)
        for partidx in range(n_parts):
            timer_sub = 0
            subiter = 0
            while timer_sub < abs(dt) and subiter < maxiter:
                if np.isnan(X_step[partidx]) or np.isnan(Y_step[partidx]) or np.isnan(Z_step[partidx]):
                    break
                X_step[partidx], Y_step[partidx], Z_step[partidx], dt_sub = find_next_position_3D(X_step[partidx], 
                                                                                                  Y_step[partidx],
                                                                                                  Z_step[partidx],  
                                                                                                  U_field_step, 
                                                                                                  V_field_step, 
                                                                                                  W_field_step, 
                                                                                                  dt_max=abs(dt) - timer_sub)
                timer_sub += dt_sub
                subiter += 1
                if subiter == maxiter:
                    print(f"Particle {partidx} reached max iter")
                    X_step[partidx] = np.nan
                    Y_step[partidx] = np.nan
                    Z_step[partidx] = np.nan
        T += dt
        if T % output_dt == 0:
            idx_out += 1
            X[idx_out, :] = X_step
            Y[idx_out, :] = Y_step
            Z[idx_out, :] = Z_step
            T_out[idx_out] = T

    if pset is None:
        return X, Y, Z, T_out
    else:
        pset.store_results(X=X, Y=Y, Z=Z, T=T_out)
        pset.add_metadata("function", "integration_loop_analytical_3D_dynamic")
        pset.add_metadata("dt", dt)
        pset.add_metadata("maxiter", maxiter)
        pset.add_metadata("T_int", T_int)
        pset.add_metadata("T0", T0) 
        pset.add_metadata("output_dt", output_dt)