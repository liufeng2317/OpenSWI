"""This file is used to calculate the dispersion curve based on the given velocity model.
The original dispersion curve forward modeling procedure is based on the CPS330 software
    https://www.eas.slu.edu/eqc/ComputerProgramsSeismology/index.html
The dispersion curve is calculated by the disba package: 
    https://github.com/keurfonluu/disba
The transform of velocity model is based on the Brocher (2005) model.
    [1] T. M. Brocher, “Empirical relations between elastic wavespeeds and density in the earth’s crust,” 
        Bull. Seismol. Soc. Amer., vol. 95, no. 6, pp. 2081–2092, Dec. 2005, doi: 10.1785/0120050077.
"""

import numpy as np
from disba import PhaseDispersion,GroupDispersion
from scipy.interpolate import interp1d

# -------------------------------------------------------
#  calculate the dispersion curve
# -------------------------------------------------------
def calculate_dispersion(vel_model,t=None,dc=0.001):
    """calculate the dispersion curve
    Args:
        vel_model: 2D numpy array
            => the velocity model (depth, vp, vs, rho)
        t: 1D numpy array
            => the periods (s)
        dc: float
            => the sampling interval (km/s) for search the dispersion curve
    Returns:
        dispersion_curve: 2D numpy array
            => the dispersion curve (period, phase velocity, group velocity)
    """
    # transform the depth to the thickness
    depth, vp, vs, rho = vel_model.T
    thickness = np.diff(depth)
    thickness = np.append(thickness, thickness[-1])
    vel_model = np.column_stack((thickness, vp, vs, rho))
    vel_model = vel_model.astype(np.float64)

    if t is None:
        # Periods must be sorted starting with low periods
        t = generate_mixed_samples(num_samples=100,start=0.2,end=10,uniform_num=50,log_num=20,random_num=30)

    # Compute the 3 first Rayleigh- and Love- wave modal dispersion curves
    try:
        pd = PhaseDispersion(*vel_model.T, dc=dc)
        gd = GroupDispersion(*vel_model.T, dc=dc)
        phase_disp = [pd(t, mode=i, wave="rayleigh") for i in range(1)]
        group_disp = [gd(t, mode=i, wave='rayleigh') for i in range(1)]
        
        phase_period,phase_vel = phase_disp[0].period,phase_disp[0].velocity
        group_period,group_vel = group_disp[0].period,group_disp[0].velocity
        
        if len(phase_period) != len(t) or len(group_period) != len(t):
            phase_period = np.zeros(len(t))
            group_period = np.zeros(len(t))
            phase_vel = np.zeros(len(t))
            group_vel = np.zeros(len(t))
            t = np.zeros(len(phase_vel))
    except Exception as e:
        # If any error occurs during computation, return arrays of zeros
        phase_vel = np.zeros(len(t))
        group_vel = np.zeros(len(t))
        t         = np.zeros(len(phase_vel))
    return np.hstack((t.reshape(-1,1),phase_vel.reshape(-1,1),group_vel.reshape(-1,1)))

# -------------------------------------------------------
#  generate the periods position
# -------------------------------------------------------
def generate_mixed_samples(num_samples, start=0.5, end=5, uniform_num=100, log_num=100, random_num=100):
    """generate periods position
    Args:
        num_samples: int
            => the number of periods
        start: float
            => the start period (s)
        end: float
            => the end period (s)
        uniform_num: int
            => the number of uniform sampling points
        log_num: int
            => the number of logarithmic sampling points
        random_num: int
            => the number of random sampling points
    Returns:
        t_final_sorted: 1D numpy array
            => the sorted periods (s)
    """
    # Uniform sampling
    t_uniform = np.linspace(start, end, num=uniform_num)
    # Logarithmic sampling
    t_log = 1/np.logspace(np.log10(1/end), np.log10(1/start), num=log_num)
    # Random sampling
    t_random = np.random.uniform(start, end, size=random_num)

    # Remove duplicates
    t_uniform_unique = np.unique(t_uniform)
    t_log_unique = np.unique(t_log)
    t_random_unique = np.unique(t_random)

    # Combine all unique sampling points
    t_combined = np.concatenate((t_uniform_unique, t_log_unique, t_random_unique))
    t_combined_unique = np.unique(t_combined)

    # Adjust final number of sampling points
    if len(t_combined_unique) > num_samples:
        t_final = np.random.choice(t_combined_unique, size=num_samples, replace=False)
    elif len(t_combined_unique) < num_samples:
        extra_samples = np.random.uniform(start, end, size=num_samples - len(t_combined_unique))
        t_final = np.concatenate((t_combined_unique, extra_samples))
        t_final = np.unique(t_final)
    else:
        t_final = t_combined_unique

    # Sort
    t_final_sorted = np.sort(t_final)

    return t_final_sorted

# -------------------------------------------------------
#  transform the velocity model (Brocher, 2005)
# -------------------------------------------------------   
def transform_vp_to_vs(vp):
    """transform the P-wave velocity to S-wave velocity (Brocher, 2005)
    Args:
        vp: 1D/2D numpy array
            => P-wave velocity (km/s)
    Returns:
        vs: 1D/2D numpy array
            => S-wave velocity (km/s)
    """
    vs = 0.7858 - 1.2344*vp + 0.7949*vp**2 - 0.1238*vp**3 + 0.0064*vp**4
    return vs

def transform_vs_to_vp(vs):
    """transform the S-wave velocity to P-wave velocity (Brocher, 2005)
    Args:
        vs: 1D/2D numpy array
            => S-wave velocity (km/s)
    Returns:
        vp: 1D/2D numpy array
            => P-wave velocity (km/s)
    """
    vp = 0.9409 + 2.0947*vs - 0.8206*vs**2+ 0.2683*vs**3 - 0.0251*vs**4
    return vp

def transform_vp_to_rho(vp):
    """transform the P-wave velocity to density (Brocher, 2005)
    Args:
        vp: 1D/2D numpy array
            => P-wave velocity (km/s)
    Returns:
        rho: 1D/2D numpy array
            => density (g/cm^3)
    """
    rho = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5
    return rho  

def transform_vs_to_vel_model(vs,depth=None):
    """transform the S-wave velocity to velocity model (Brocher, 2005)
    Args:
        vs: 1D numpy array
            => S-wave velocity (km/s)
        depth: 1D numpy array
            => depth (km)
    Returns:
        vel_model: 2D numpy array
            => the velocity model (depth, vp, vs, rho)
    """
    if depth is None:
        depth       = np.arange(0, vs.shape[-1])*0.04
    vp          = transform_vs_to_vp(vs)

    # depth > 120 km, the P-wave velocity is equal to the S-wave velocity * 1.79
    mask = depth>120
    vp[mask] = vs[mask]*1.79
    rho         = transform_vp_to_rho(vp)
    vel_model   = np.column_stack((depth, vp, vs, rho))
    return vel_model

def transform_vp_to_vel_model(vp,depth=None):
    """get the velocity model (Brocher, 2005)
    Args:
        vp: 1D numpy array
            => P-wave velocity (km/s)
        depth: 1D numpy array
            => depth (km)
    Returns:
        vel_model: 2D numpy array
            => the velocity model (depth, vp, vs, rho)
    """ 
    if depth is None:
        depth       = np.arange(0, vp.shape[-1])*0.04
    vs          = transform_vp_to_vs(vp)
    rho         = transform_vp_to_rho(vp)
    vel_model   = np.column_stack((depth, vp, vs, rho))
    return vel_model

# -------------------------------------------------------
#  interpolate the velocity model
# -------------------------------------------------------
def interpolate_vel_model(vel_model,depth_new,interp_method="nearest"):
    """interpolate the velocity model
    Args:
        vel_model: 2D numpy array
            => the velocity model (depth, vp, vs, rho)
        depth_new: 1D numpy array
            => the new depth (km)
        interp_method: str
            => the interpolation method
    Returns:
        vel_model: 2D numpy array
            => the velocity model (depth, vp, vs, rho)
    """
    depth,vp,vs,rho = vel_model.T
    f_vp  = interp1d(depth,vp,kind=interp_method,fill_value=vp[-1],bounds_error=False)
    f_vs  = interp1d(depth,vs,kind=interp_method,fill_value=vs[-1],bounds_error=False)
    f_rho = interp1d(depth,rho,kind=interp_method,fill_value=rho[-1],bounds_error=False)
    vp_new = f_vp(depth_new)
    vs_new = f_vs(depth_new)
    rho_new = f_rho(depth_new)
    vel_model_new = np.column_stack((depth_new,vp_new,vs_new,rho_new))
    return vel_model_new

# -------------------------------------------------------
#  generate the initial model based on empirical formula 
# -------------------------------------------------------
def gen_init_model(t,cg_obs,thick,area=False):
    """
    generate the initial model based on empirical formula 
    developed by Thomas M.Brocher (2005).
    ---------------------
    Input Parameters:
        t : 1D numpy array 
            => period of observaton dispersion points
        cg_obs: 1D numpy array 
            => phase velocity of observation dispersion points
        thick : 1D numpy array 
            => thickness of each layer
    Output: the initialize model
        thick : 1D numpy array 
            => thickness
        vs : 1D numpy array 
            => the shear wave velocity
        vp : 1D numpy array 
            => the compress wave velocity
        rho: 1D numpy array 
            => the density
    --------------------
    Output parameters:
        model:Dict 
            => the generated model
    """
    wavelength  = t*cg_obs
    nlayer      = len(thick)
    lambda2L    = 0.65      # the depth faction 0.63L
    beta        = 0.92      # the poisson's ratio
    eqv_lambda = lambda2L*wavelength
    lay_model = np.zeros((nlayer,2))
    lay_model[:,0] = thick
    for i in range(nlayer-1):
        if i == 0:
            up_bound = 0
        else:
            up_bound = up_bound + lay_model[i-1,0] # the top-layer's depth
        low_bound = up_bound + lay_model[i,0] # the botton-layer's depth
        # vs for every layer
        lambda_idx = np.argwhere((eqv_lambda>up_bound) & (eqv_lambda<low_bound))
        if len(lambda_idx)>0:
            lay_model[i,1] = np.max(cg_obs[lambda_idx])/beta # phase velocity -> vs
        else:
            lambda_idx = np.argmin(np.abs(eqv_lambda - low_bound))
            lay_model[i,1] = cg_obs[lambda_idx]/beta
    # set the last layer
    lay_model[nlayer-1,0] = 0
    lay_model[nlayer-1,1] = np.max(cg_obs)*1.1
    thick = lay_model[:,0]
    vs = lay_model[:,1]
    vp = transform_vs_to_vp(vs)
    depth = np.cumsum(thick)
    
    mask = depth>120
    vp[mask] = vs[mask]*1.79
    rho = transform_vp_to_rho(vp)
    
    model = {
        "depth":depth,
        "vp":vp,
        "vs":vs,
        "rho":rho
    }
    if area:
        return np.column_stack((depth,vp,vs,rho)) 
    else:
        return model

# -------------------------------------------------------
#  generate the velocity model from the S-wave velocity
# -------------------------------------------------------
def gen_model_from_vs(depth,vs,area=False,Brocher=True):
    """
    generate the initial model based on empirical formula 
    developed by Thomas M.Brocher (2005).
    ---------------------
    Input Parameters:
        thick : Array(1D) 
            => the thickness of layer 
        vs    : Array(1D)
            => the shear wave velocity
        area  : boolen 
            => the output format
    --------------------
    Output parameters:
        model:Dict 
            the generated model
    """
    depth       = np.array(depth)
    thickness   = np.diff(depth)
    thickness   = np.insert(thickness,-1,thickness[-1]) 
    vs = np.array(vs)
    if Brocher:
        vp = transform_vs_to_vp(vs)
    else:
        vp = 1.79*vs
    mask = depth>120
    vp[mask] = vs[mask]*1.79
    rho = transform_vp_to_rho(vp)
    model = {
        "depth":depth,
        "vp":vp,
        "vs":vs,
        "rho":rho
    }
    if area:
        return np.column_stack((depth, vp, vs, rho))
    else:
        return model
    
# -------------------------------------------------------
#  smooth the data
# -------------------------------------------------------
def smooth_data(y, window_size=7):
    """
    Smooth the input data using a moving average filter, with padding to reduce edge effects.
    
    Parameters:
    y (array-like): The input data to be smoothed.
    window_size (int): Size of the moving average window (default is 7).
    
    Returns:
    y_smooth (array-like): Smoothed data.
    """

    # Save the first data point to preserve it after smoothing
    first_point = y[0]

    # Create a moving average window (a window of ones normalized by the window size)
    window = np.ones(window_size) / window_size

    # Pad the input data using 'reflect' mode to minimize edge effects during convolution
    y_padded = np.pad(y, (window_size // 2, window_size // 2), mode='reflect')

    # Apply the convolution between the padded data and the smoothing window
    y_smooth = np.convolve(y_padded, window, mode='valid')

    # Replace the first point of the smoothed data with the original first point
    y_smooth[0] = first_point
    
    return y_smooth