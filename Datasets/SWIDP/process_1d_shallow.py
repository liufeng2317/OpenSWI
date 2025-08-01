import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, PchipInterpolator


# -------------------------------------------------------
#  combine the velocity profiles with the same S-wave velocity
# -------------------------------------------------------
def combine_same_vs(vs,depth=None,vel_threshold=0.01):
    """combine the velocity profiles with the same S-wave velocity
    Args:
        vs: 1D numpy array
            => S-wave velocity (km/s)
        depth: 1D numpy array
            => depth (km)
    Returns:
        new_vs: 1D numpy array
            => S-wave velocity (km/s)
        new_depth: 1D numpy array
            => depth (km)
    """
    if depth is None:
        depth = np.arange(70)*0.04
    
    new_vs,new_depth = [],[]
    i = 0
    while i < len(vs)-1:
        # Find the next layer with different velocity
        next_i = i + 1
        while next_i < len(vs) and np.abs(vs[next_i] - vs[i]) < vel_threshold:
            next_i += 1
        # Add the current layer to new model
        new_depth.append(depth[i])
        new_vs.append(vs[i])
        # Move to next different layer
        i = next_i
    # add the last layer
    new_depth.append(depth[-1])
    new_vs.append(vs[-1])
    return np.array(new_vs),np.array(new_depth)


# -------------------------------------------------------
#  remove the thin layers
# -------------------------------------------------------
def remove_thin_layer(vs, depth, thickness_threshold=0.1):
    """ remove thin layers
    """
    depth = np.array(depth)
    thickness = np.diff(depth)
        
    select_idx = np.where(thickness > thickness_threshold)[0]
    
    # nodes are the depth of the selected thickness
    if len(thickness) - 1 in select_idx:
        depth_selected = np.concatenate(([depth[0]], depth[select_idx + 1]))
        vs_selected = np.concatenate(([vs[0]], vs[select_idx + 1]))
    else:
        depth_selected = np.concatenate(([depth[0]], depth[select_idx + 1], [depth[-1]]))
        vs_selected = np.concatenate(([vs[0]], vs[select_idx + 1], [vs[-1]]))
    return vs_selected,depth_selected

# -------------------------------------------------------
#  perturb the velocity and depth
# -------------------------------------------------------
def perturb_vs_depth(vs,depth,
                     vs_perturbation=0.05,
                     thickness_perturbation=0.1,
                     vel_threshold=0.1):
    """perturb the velocity and depth
    Args:
        vs: 1D numpy array
            => S-wave velocity (km/s)
        depth: 1D numpy array
            => depth (km)
        vs_perturbation: float
            => perturbation of the velocity (km/s)
        thickness_perturbation: float
            => perturbation of the thickness (km)
        vel_threshold: float
            => minimum thickness (km)
    Returns:
        new_vs: 1D numpy array
            => S-wave velocity (km/s)
        new_depth: 1D numpy array
            => depth (km)
    """
    # Add random perturbation while maintaining relative velocity relationships
    new_vs = vs * (1 + np.random.normal(0,vs_perturbation,vs.shape)) 
    
    # pertubation the thickness -> reconstruct the depth
    new_thickness = np.insert(np.diff(depth),0,0)
    new_thickness[1:] = new_thickness[1:] * (1 + np.random.normal(0,thickness_perturbation,new_thickness[1:].shape))
    new_depth = np.cumsum(new_thickness)
    new_depth[-1] = depth[-1].copy()
    
    # combine the thin layers with thickness < vel_threshold
    new_vs,new_depth = combine_same_vs(new_vs,new_depth,vel_threshold=vel_threshold)
    
    # ensure vs[-1] > vs[-2]
    # if new_vs[-1] < new_vs[-2]:
    #     new_vs[-1] = new_vs[-2] *(1+abs(np.random.normal(0,0.05)))  # Ensure strictly greater than
    
    return new_vs,new_depth

def combine_thin_sandwich(vs, depth, thickness_threshold=0.1, uniform_thickness=0.04, gradient_threshold=0.005, return_idx=False):
    """ Remove anomalous thin sandwich layers (high or low velocity layers)
    Args:
        vs: 1D numpy array
            => S-wave velocity (km/s)
        depth: 1D numpy array
            => depth (km)
        thickness_threshold: float
            => minimum layer thickness to keep
        uniform_thickness: float
            => uniform thickness (km)
        gradient_threshold: float
            => maximum velocity difference ratio to consider anomalous
        return_idx: bool
            => return the index of the sandwich layers
    Returns:
        new_vs: 1D numpy array
            => S-wave velocity (km/s) with anomalous layers removed
        sandwich_idx (optional): list
            => index of the sandwich layers [start_idx,last_idx]
    """
    # max number of layers to check
    max_layers = int(thickness_threshold/uniform_thickness)

    # Calculate layer thicknesses and velocity gradients
    gradients = np.diff(vs) / np.diff(depth)
    gradients_signs = np.sign(gradients)

    # Initialize output arrays
    new_vs = vs.copy()

    # Step 1: Find layers with large gradients
    large_grad_idx = np.where(np.abs(gradients) > gradient_threshold)[0]
    
    sandwich_idx = []
    # Step 2 & 3: Check for sandwich structures and merge layers
    for i in range(len(large_grad_idx)):
        # get the current gradient sign
        start_idx = large_grad_idx[i]
        grad_sign = np.sign(gradients[start_idx])

        # get the next N layers signs
        next_n_layers_signs = gradients_signs[start_idx+1:start_idx+max_layers]

        # Check if gradients have opposite signs and larger than gradient_threshold
        metric1 = next_n_layers_signs == -grad_sign
        metric2 = np.abs(gradients[start_idx+1:start_idx+max_layers]) > gradient_threshold
        metric = metric1*metric2
        if np.any(metric):
            # get the first true index
            last_idx = np.where(metric)[0][0] + start_idx + 1
            if last_idx > 0:
                # Merge by first velocity
                new_vs[start_idx:last_idx+1] = vs[start_idx]
            sandwich_idx.append([start_idx,last_idx])
    if return_idx:
        return sandwich_idx, new_vs
    else:
        return new_vs

# -------------------------------------------------------
#  smooth the velocity by B-spline interpolation
# -------------------------------------------------------
def smooth_vs_by_node_interp(vs, depth, n_nodes=10, method="pchip"):
    """ Smooth the velocity by B-spline interpolation
    Args:
        vs: 1D numpy array
            => S-wave velocity (km/s)
        depth: 1D numpy array
            => depth (km)
        n_nodes: int
            => number of nodes
        method: str
            => method to smooth the velocity
    Returns:
        vs_smooth: 1D numpy array
            => smoothed velocity (km/s)
    """
    depth = np.array(depth)
    nodes = np.linspace(depth.min(), depth.max(), n_nodes)
    vs_nodes = np.interp(nodes, depth, vs)
    if method == "spline":
        spline = make_interp_spline(nodes, vs_nodes, k=3)
        return spline(depth)
    elif method == "pchip":
        pchip = PchipInterpolator(nodes, vs_nodes)
        return pchip(depth)
    else:
        raise ValueError(f"Invalid method: {method}")
    
def augment_workflow(vs,depth,
                     perturb_num = 100,
                     vs_perturbation=0.05,
                     thickness_perturbation=0.1,
                     vel_threshold=0.01,
                     thickness_threshold = 0.01,
                     min_layers_num=3,
                     smooth_vel=False,
                     smooth_nodes=10,
                     ):
    """perturb the velocity and depth
    Args:
        vs: 1D numpy array
            => S-wave velocity (km/s)
        depth: 1D numpy array
            => depth (km)
        perturb_num: int
            => number of perturbations
        vs_perturbation: float
            => perturbation of the velocity (km/s)
        thickness_perturbation: float
            => perturbation of the thickness (km)
        vel_threshold: float
            => minimum thickness (km)
        min_layers_num: int
            => minimum number of layers
    Returns:
        perturb_vs: 2D numpy array
            => perturbed velocity (km/s)
    """
    # step 1: combine the same vs value
    new_vs,new_depth = combine_same_vs(vs,depth,vel_threshold=vel_threshold)

    # step 2: remove the thin layers with thickness < vel_threshold
    new_vs,new_depth = remove_thin_layer(new_vs,new_depth,thickness_threshold=thickness_threshold)
    
    # step 3: if the number of layers is less than min_layers_num, return 0
    if len(new_depth) < min_layers_num:
        return np.zeros((perturb_num,len(depth)))
    
    # step 4: perturb the velocity and depth
    perturb_vs = []
    for i in range(perturb_num):
        if i == 0:
            temp_vs = new_vs
            temp_depth = new_depth
        else:
            # Add random perturbation while maintaining relative velocity relationships
            temp_vs,temp_depth = perturb_vs_depth(new_vs,new_depth,vs_perturbation,thickness_perturbation,vel_threshold)

        # interpolate the new_depth and new_vs to the original depths
        f = interp1d(temp_depth,temp_vs,kind="previous")
        interp_vs = f(depth)
        
        # step 5: smooth the velocity
        if smooth_vel:
            interp_vs = smooth_vs_by_node_interp(interp_vs,depth,n_nodes=smooth_nodes,method="pchip")
        perturb_vs.append(interp_vs)
        
    perturb_vs = np.array(perturb_vs)
    return perturb_vs