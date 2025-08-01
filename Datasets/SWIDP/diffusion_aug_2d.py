import numpy as np
from SWIDP.process_1d_shallow import combine_same_vs, perturb_vs_depth, smooth_vs_by_node_interp
from scipy.interpolate import interp1d

# -------------------------------------------------------
#  extract the 1D profile from the 2D model
# -------------------------------------------------------
def extract_1d_profile(vs_model,station_idx=None,method="mean"):
    """extract the 1D profile from the 2D model
    Args:
        vs_model: 2D array [n_depth,n_x]
            => shear velocity model
        method: str
            => method to extract the 1D profile
    Returns:
        vs_profile: 2D array [n_sta,n_depth]
            => shear velocity profile
    """    
    if method == "mean":
        vs_profile = np.mean(vs_model,axis=1)
        vs_profile = vs_profile.reshape(1,-1)
    elif method == "st" and isinstance(station_idx,int): # single station
        vs_profile = vs_model[:,station_idx]
        vs_profile = vs_profile.reshape(1,-1)
    elif method == "mt" and isinstance(station_idx,list):
        vs_profile = vs_model[:,station_idx]
        vs_profile = vs_profile.T
    elif method == "random": # random station
        vs_profile = vs_model[:,np.random.randint(0,vs_model.shape[1])]
        vs_profile = vs_profile.reshape(1,-1)
    elif method == "all": # all stations
        vs_profile = vs_model
        vs_profile = vs_profile.T
    else:
        raise ValueError(f"Invalid method: {method}")
    return vs_profile


# -------------------------------------------------------
#  normalize & denormalize the shear velocity
# -------------------------------------------------------
def max_min_normalize(vs):
    """Normalize shear velocity.
    Args:
        vs: 1D array
            => shear velocity (km/s)
    Returns:
        vs: 1D array
            => normalized shear velocity (0-1)
    """
    vs = np.array(vs)
    vs = (vs-np.min(vs))/(np.max(vs)-np.min(vs))
    return vs

def max_min_denormalize(vs, vmin=0.2, vmax=3.2):
    """Denormalize shear velocity.
    Args:
        vs: 1D array
            => normalized shear velocity (0-1)
        vmin: float,
            => minimum shear velocity (km/s)
        vmax: float,
            => maximum shear velocity (km/s)
    Returns:
        vs: 1D array
            => shear velocity (km/s)
    """
    vs = np.array(vs)
    # make sure the vs is normalized
    vs = max_min_normalize(vs)

    # denormalize to [vmin,vmax]
    vs = vs*(vmax-vmin)+vmin
    return vs

# -------------------------------------------------------
#  interpolate the shear velocity
# -------------------------------------------------------
def interpolate_vs(vs,depth,depth_interp,kind="previous"):
    """interpolate the shear velocity
    Args:
        vs: 1D array
            => shear velocity (km/s)
    """
    # padding the last layer of depth and vs larger than depth_interp
    eps = 0.01
    if depth.max() < depth_interp.max():
        vs = np.insert(vs,len(vs),vs[-1])
        depth = np.insert(depth,len(depth),depth_interp[-1]+eps)
    elif depth.min() > depth_interp.min():
        vs = np.insert(vs,0,vs[0])
        depth = np.insert(depth,0,depth[0]-eps)
    
    # interpolate the shear velocity
    f = interp1d(depth,vs,kind=kind)
    vs_interp = f(depth_interp)
    return vs_interp

# -------------------------------------------------------
#  extract 1D profile and interpolation
# -------------------------------------------------------

def diffusion_extract_process(vs_model,
                              extract_station_idx=None,
                              extract_method="mean",
                              denorm_vmin_range=[0.2,0.5],
                              denorm_vmax_range=[2.5,3.5],
                              combine_vel_threshold=0.1,
                              interp_depth=None,
                              interp_kind="previous",
                              smooth_vel=False,
                              smooth_nodes=10,
                              smooth_kind="pchip",
                              ):
    """diffusion extract process
    Args:
        vs_model: 2D array [n_depth,n_x]
            => shear velocity model
        extract_station_idx: int,
            => station index
        extract_method: str
            => method to extract the 1D profile
        denorm_vmin: float
            => minimum shear velocity (km/s)
        denorm_vmax: float
            => maximum shear velocity (km/s)
        combine_vel_threshold: float
            => velocity threshold for combining the same vs
        interp_depth: 1D array
            => target depth for interpolation
        interp_kind: str
            => kind of interpolation
        smooth_vel: bool
            => whether to smooth the vs
        smooth_nodes: int
            => number of nodes for smoothing
        smooth_kind: str
            => kind of smoothing
    Returns:
        vs_aug: 2D array [n_sta,n_depth]
            => extracted shear velocity
        depth_aug: 2D array [n_sta,n_depth]
            => extracted depth
    """
    # Step 1: extract the 1D profile
    vs_profiles = extract_1d_profile(vs_model,extract_station_idx,extract_method)
    depth = np.arange(vs_profiles.shape[-1])*0.04

    if interp_depth is None:
        interp_depth = depth

    vs_aug,depth_aug = [],[]

    for i in range(vs_profiles.shape[0]):
        vs_profile = vs_profiles[i,:]

        # Step 2: de-normalize the shear velocity
        denorm_vmin = np.random.uniform(denorm_vmin_range[0],denorm_vmin_range[1])
        denorm_vmax = np.random.uniform(denorm_vmax_range[0],denorm_vmax_range[1])
        vs_denormed = max_min_denormalize(vs_profile,denorm_vmin,denorm_vmax)

        # Step 3: combine the same vs
        vs_combined,depth_combined = combine_same_vs(vs_denormed,depth,combine_vel_threshold)

        # step 4: smooth the vs
        if smooth_vel:
            vs_smooth = smooth_vs_by_node_interp(vs_combined,depth_combined,n_nodes=smooth_nodes,method=smooth_kind)
        else:
            vs_smooth = vs_combined

        # Step 5: interpolate the shear velocity
        vs_interpolated = interpolate_vs(vs_smooth,depth_combined,interp_depth,interp_kind)

        vs_aug.extend(vs_interpolated)
        depth_aug.extend(interp_depth)

    vs_aug = np.array(vs_aug).reshape(-1,len(interp_depth)) # [n_sta,n_depth]
    depth_aug = np.array(depth_aug).reshape(-1,len(interp_depth)) # [n_sta,n_depth]
    return vs_aug,depth_aug

# -------------------------------------------------------
#  augmentation 1D profile
# -------------------------------------------------------

def diffusion_aug_extract_process(vs_model,
                                  extract_station_idx=None,
                                  extract_method="mean",
                                  denorm_vmin_range=[0.2,0.5],
                                  denorm_vmax_range=[2.5,3.5],
                                  combine_vel_threshold=0.1,
                                  interp_depth=np.arange(70)*0.04,
                                  interp_kind="previous",
                                  smooth_vel=False,
                                  smooth_nodes=10,
                                  smooth_kind="pchip",
                                  aug_nums = 100,
                                  aug_vs_probility=0.05,
                                  aug_thickness_probility=0.1,
                                  aug_vel_threshold=0.1,
                                  smooth_aug_vel=False,
                                  smooth_aug_nodes=10,
                                  smooth_aug_kind="pchip",
                                  ):
    """diffusion augmentation extract process
    Args:
        vs_model: 2D array [n_depth,n_x]
            => shear velocity model
        extract_station_idx: int,
            => station index
        extract_method: str
            => method to extract the 1D profile
        denorm_vmin: float
            => minimum shear velocity (km/s)
        denorm_vmax: float
            => maximum shear velocity (km/s)
        combine_vel_threshold: float
            => velocity threshold for combining the same vs
        interp_depth: 1D array
            => target depth for interpolation
        interp_kind: str
            => kind of interpolation
        smooth_vel: bool
            => whether to smooth the vs
        smooth_nodes: int
            => number of nodes for smoothing
        smooth_kind: str
            => kind of smoothing
        aug_nums: int
            => number of augmentation
        aug_vs_probility: float
            => probability of vs perturbation
        aug_thickness_probility: float
            => probability of thickness perturbation
        aug_vel_threshold: float
            => velocity threshold for removing the thin layers
        aug_min_layers_num: int
            => minimum number of layers
        smooth_aug_vel: bool
            => whether to smooth the augmented vs
        smooth_aug_nodes: int
            => number of nodes for smoothing the augmented vs
        smooth_aug_kind: str
            => kind of smoothing the augmented vs
    Returns:
        vs_aug: 3D array [n_sta,n_aug,n_depth]
            => augmented shear velocity
        depth_aug: 3D array [n_sta,n_aug,n_depth]
            => augmented depth
    """
    # Step 1: extract the 1D profile
    vs_profiles = extract_1d_profile(vs_model,extract_station_idx,extract_method)
    depth = np.arange(vs_profiles.shape[-1])*0.04

    vs_aug,depth_aug = [],[]

    for i in range(vs_profiles.shape[0]):
        vs_profile = vs_profiles[i,:]

        # Step 2: de-normalize the shear velocity
        denorm_vmin = np.random.uniform(denorm_vmin_range[0],denorm_vmin_range[1])
        denorm_vmax = np.random.uniform(denorm_vmax_range[0],denorm_vmax_range[1])
        vs_denormed = max_min_denormalize(vs_profile,denorm_vmin,denorm_vmax)

        # Step 3: combine the same vs
        vs_combined,depth_combined = combine_same_vs(vs_denormed,depth,combine_vel_threshold)

        # step 4: smooth the vs
        if smooth_vel:
            vs_smooth = smooth_vs_by_node_interp(vs_combined,depth_combined,n_nodes=smooth_nodes,method=smooth_kind)
        else:
            vs_smooth = vs_combined

        # Step 5: augmentation 1D profile
        vs_aug_st,depth_aug_st = [],[]
        for j in range(aug_nums):
            if j == 0:
                vs_aug_1d = vs_smooth
                depth_aug_1d = depth_combined
            else:
                vs_aug_1d,depth_aug_1d = perturb_vs_depth(vs_smooth,depth_combined,
                                                        vs_perturbation=aug_vs_probility,
                                                        thickness_perturbation=aug_thickness_probility,
                                                        vel_threshold=aug_vel_threshold)
            
            # Step 6: interpolate the shear velocity
            vs_interpolated = interpolate_vs(vs_aug_1d,depth_aug_1d,interp_depth,interp_kind)

            # step 7: smooth the augmented shear velocity
            if smooth_aug_vel:
                vs_interpolated = smooth_vs_by_node_interp(vs_interpolated,interp_depth,n_nodes=smooth_aug_nodes,method=smooth_aug_kind)
            vs_aug_st.append(vs_interpolated)
            depth_aug_st.append(interp_depth)
        vs_aug.append(np.array(vs_aug_st))
        depth_aug.append(np.array(depth_aug_st))

    vs_aug = np.array(vs_aug) # [n_sta,n_aug,n_depth]
    depth_aug = np.array(depth_aug) # [n_sta,n_aug,n_depth]
    return vs_aug,depth_aug