import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, PchipInterpolator
from SWIDP.dispersion import *

# -------------------------------------------------------
#  find the moho depth
# -------------------------------------------------------
def find_moho_depth(vs,depth,
                    moho_depth_range=[5,90],
                    gradient_search = True,
                    gradient_threshold=0.1,
                    gradient_search_range=10):
    """find the moho depth
    Args:
        vs: 1D numpy array
            => S-wave velocity (km/s)
        depth: 1D numpy array
            => depth (km)
        moho_depth_range: list
            => moho depth range (km)
        gradient_search: bool
            => whether to search by gradient
        gradient_threshold: float
            => gradient threshold
        gradient_search_range: int
            => gradient search range (km)    Returns:
        moho_depth_idx: int
            => moho depth index
    """
    # first find by gradient and max value
    moho_depth_idx_max = find_moho_depth_by_max_value(vs,depth,moho_depth_range)
    if gradient_search:
        moho_depth_idx_gradient = find_moho_depth_by_gradient(vs,depth,gradient_threshold,moho_depth_range)
        if moho_depth_idx_gradient is None:
            return moho_depth_idx_max
        # if the difference between gradient and max value is less than 10, return the max value
        if np.abs(moho_depth_idx_gradient-moho_depth_idx_max) < gradient_search_range:
            return moho_depth_idx_max
        else:
            moho_depth_idx_max = find_moho_depth_by_max_value(vs,depth,[moho_depth_idx_gradient-gradient_search_range,moho_depth_idx_gradient+gradient_search_range])
            return moho_depth_idx_max
    else:
        return moho_depth_idx_max
    

def find_moho_depth_by_max_value(vs,depth,moho_depth_range=[5,90]):
    """find the moho depth
    Args:
        vs: 1D numpy array
            => S-wave velocity (km/s)
        depth: 1D numpy array
            => depth (km)
        moho_depth_range: list
            => moho depth range (km)
    Returns:
        moho_depth_idx: int
            => moho depth index
    """
    # find the min and max depth index
    min_depth_idx = np.argmin(np.abs(depth-moho_depth_range[0]))
    max_depth_idx = np.argmin(np.abs(depth-moho_depth_range[1]))
    # find the max velocity index in the moho depth range
    moho_depth_idx = np.argmax(vs[min_depth_idx:max_depth_idx])+min_depth_idx
    return moho_depth_idx

def find_moho_depth_by_gradient(vs,depth,gradient_threshold=0.1,moho_depth_range=[5,90]):
    """find the moho depth by gradient
    Args:
        vs: 1D numpy array
            => S-wave velocity (km/s)
        depth: 1D numpy array
            => depth (km)
        gradient_threshold: float
            => gradient threshold
        moho_depth_range: list
            => moho depth range (km)
    Returns:
        moho_depth_idx: int
            => moho depth index
    """
    moho_min_vs = 3.2 # the minimum vs for moho
    
    # calculate the gradient
    gradients = np.diff(vs) / np.diff(depth)

    # gradient normalize to [-1,1]
    gradients = 2 * (gradients - gradients.min()) / (gradients.max() - gradients.min()) - 1
    
    # find the larger than gradient_threshold index (which is >0)
    moho_depth_idx = np.argwhere(gradients>gradient_threshold)[:,0]

    # find the max gradient index in the moho depth range
    min_depth_idx = np.argmin(np.abs(depth-moho_depth_range[0]))
    max_depth_idx = np.argmin(np.abs(depth-moho_depth_range[1]))

    # find the max gradient index in the moho depth range
    moho_depth_idx = moho_depth_idx[(moho_depth_idx >= min_depth_idx) & (moho_depth_idx <= max_depth_idx) & (vs[moho_depth_idx] >= moho_min_vs)]
    if len(moho_depth_idx) > 0:
        moho_depth_idx = moho_depth_idx[-1]+1
    else:
        moho_depth_idx = None
    return moho_depth_idx

# -------------------------------------------------------
#  perturb the moho depth and velocity
# -------------------------------------------------------
def perturb_moho_vs(vs, depth, moho_idx, 
                        moho_perturb_range=[-10,10], 
                        vs_perturb_range=[-0.1,0.1], 
                        moho_idx_range=[5,95],
                        crust_nodes=6,
                        mantle_nodes=12,
                        moho_shift_range=10,
                        return_nodes=False):
    """Perturb Moho depth and velocity while keeping other parts linear interpolated
    
    Args:
        vs: 1D numpy array
            => S-wave velocity profile (km/s)
        depth: 1D numpy array  
            => depth profile (km)
        moho_idx: int
            => index of Moho discontinuity
        moho_perturb_range: float
            => perturbation range for Moho depth (±fraction) (default: 10)
        vs_perturb_range: float
            => perturbation range for velocity (±fraction) (default: 0.2)
        moho_idx_range: list
            => range of Moho depth index (default: [5,95])
        return_nodes: bool
            => whether to return the control points (default: False)
        crust_nodes: int
            => number of crust nodes (default: 6)
        mantle_nodes: int
            => number of mantle nodes (default: 12)
        moho_shift_range: int
            => perturbation range for Moho depth (±integer) (default: 10)
            
    Returns:
        vs_perturbed: 1D numpy array
            => perturbed velocity profile
    """
    # Copy original profile
    vs_perturbed = vs.copy()
    
    # Randomly perturb Moho depth within range
    moho_shift = np.random.randint(moho_perturb_range[0],moho_perturb_range[1])
    new_moho_idx = moho_idx + moho_shift
    new_moho_idx = np.clip(new_moho_idx, moho_idx_range[0], moho_idx_range[1])
    moho_shift_range = np.clip(moho_shift_range,1,new_moho_idx)

    # Perturb velocity at new Moho
    vs_shift = np.random.uniform(vs_perturb_range[0], vs_perturb_range[1])
    new_moho_vs = vs[moho_idx] + vs_shift
    
    # Select X nodes before original Moho and Y nodes after original Moho as control points
    crust_interval = int(np.ceil(moho_idx/crust_nodes))
    mantle_interval = int(np.ceil((len(vs) - moho_idx)/mantle_nodes))
    
    # calculate the control points (crust)
    depth_crust = depth[:moho_idx][::crust_interval]
    vs_crust = vs[:moho_idx][::crust_interval]
    # select the nodes < new moho depth & vs < new moho vs
    cond1 = depth_crust < depth[new_moho_idx-moho_shift_range]
    cond2 = vs_crust < new_moho_vs
    cond = cond1*cond2
    # make sure at least 1 node is selected
    if np.sum(cond) == 0:
        cond[0] = True
    depth_crust_select = depth_crust[cond]
    vs_crust_select = vs_crust[cond]


    # calculate the control points (mantle)
    depth_mantle = depth[moho_idx:][::mantle_interval]
    vs_mantle = vs[moho_idx:][::mantle_interval]
    # select the nodes > new moho depth
    cond3 = depth_mantle > depth[new_moho_idx+moho_shift_range]
    depth_mantle_select = depth_mantle[cond3]
    vs_mantle_select = vs_mantle[cond3]

    # Connect crust and mantle nodes for spline interpolation
    controle_node_depth = np.concatenate([depth_crust_select,[depth[new_moho_idx]],depth_mantle_select])
    controle_node_vs = np.concatenate([vs_crust_select,[new_moho_vs],vs_mantle_select])

    # spl_crust_new = make_interp_spline(controle_node_depth, controle_node_vs, k=3)  # k=3 for cubic splines
    # vs_perturbed = spl_crust_new(depth)    
    spl_crust_new = PchipInterpolator(controle_node_depth, controle_node_vs)
    vs_perturbed = spl_crust_new(depth)
    
    if return_nodes:
        return vs_perturbed, controle_node_depth, controle_node_vs
    else:
        return vs_perturbed

from scipy.ndimage import gaussian_filter1d
def augment_crust_moho_mantle(vs, depth, moho_idx, 
                        vs_perturb_range=[-0.1,0.1], 
                        crust_nodes_range=[3,10],
                        mantle_nodes_range=[6,12],
                        moho_shift_range=10,
                        gaussian_smooth_sigma=2,
                        return_nodes=False,
                        random_seed=None):
    """Perturb all the profile [crust,moho,mantle]: make sure the same moho position
    
    Args:
        vs: 1D numpy array
            => S-wave velocity profile (km/s)
        depth: 1D numpy array  
            => depth profile (km)
        moho_idx: int
            => index of Moho discontinuity
        vs_perturb_range: float
            => perturbation range for velocity (±fraction) (default: 0.2)
        crust_nodes_range: list
            => range of number of crust nodes (default: [3,10])
        mantle_nodes_range: list
            => range of number of mantle nodes (default: [6,12])
        moho_shift_range: int
            => perturbation range for Moho depth (±integer) (default: 10)
        gaussian_smooth_sigma: float
            => smoothing parameter for gaussian filter (default: 2)
        return_nodes: bool
            => whether to return the control points (default: False)
    Returns:
        vs_perturbed: 1D numpy array
            => perturbed velocity profile
        controle_node_depth: 1D numpy array
            => control points depth
        controle_node_vs: 1D numpy array
            => control points velocity
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Copy original profile
    vs_perturbed = vs.copy()
    crust_nodes  = np.random.randint(crust_nodes_range[0],crust_nodes_range[1])
    mantle_nodes = np.random.randint(mantle_nodes_range[0],mantle_nodes_range[1])
    
    moho_shift_range = np.min([moho_shift_range,moho_idx-1])
    
    #----------------------------------------------------
    #  perturb the moho depth and velocity
    #----------------------------------------------------
    # Perturb velocity at new Moho
    vs_shift = np.random.uniform(vs_perturb_range[0], vs_perturb_range[1])
    new_moho_vs = vs[moho_idx] + vs_shift
    
    #----------------------------------------------------
    #  calculate the control points
    #----------------------------------------------------
    # Select X nodes before original Moho and Y nodes after original Moho as control points
    crust_interval = int(np.ceil(moho_idx/crust_nodes))
    crust_interval = np.clip(crust_interval,1,5)
    mantle_interval = int(np.ceil((len(vs) - moho_idx)/mantle_nodes))
    mantle_interval = np.clip(mantle_interval,1,20)
    
    # calculate the control points (crust)
    depth_crust = depth[:moho_idx][::crust_interval]
    vs_crust = vs[:moho_idx][::crust_interval]
    # select the nodes < new moho depth & vs < new moho vs
    cond1 = depth_crust < depth[moho_idx-moho_shift_range]
    cond2 = vs_crust < new_moho_vs
    cond = cond1*cond2
    # make sure at least 1 node is selected
    if np.sum(cond) == 0:
        cond[0] = True
    depth_crust_select = depth_crust[cond]
    vs_crust_select = vs_crust[cond]
    # if the first node is not the same as the original, then add the original first node
    if vs_crust_select[0] != vs[0]:
        vs_crust_select = np.insert(vs_crust_select,0,vs[0])
        depth_crust_select = np.insert(depth_crust_select,0,depth[0])

    # calculate the control points (mantle)
    depth_mantle = depth[moho_idx:][::mantle_interval]
    vs_mantle = vs[moho_idx:][::mantle_interval]
    # select the nodes > new moho depth
    cond3 = depth_mantle > depth[moho_idx+moho_shift_range]
    depth_mantle_select = depth_mantle[cond3]
    vs_mantle_select = vs_mantle[cond3]
    # if the last node is not the same as the original, then add the original last node
    if vs_mantle_select[-1] != vs[-1]:
        vs_mantle_select = np.append(vs_mantle_select,vs[-1])
        depth_mantle_select = np.append(depth_mantle_select,depth[-1])

    #----------------------------------------------------
    # random perturb the crust & mantle nodes
    #----------------------------------------------------
    # crust
    vs_crust_select[1:] = vs_crust_select[1:] + np.random.uniform(vs_perturb_range[0], vs_perturb_range[1], len(vs_crust_select)-1)
    # Smooth crust nodes
    if len(vs_crust_select) >= 2:
        vs_crust_select = gaussian_filter1d(vs_crust_select, sigma=gaussian_smooth_sigma)
    # not change the first node
    vs_crust_select[0] = vs[0]
    # remove the same depth nodes
    unique_depths, unique_indices = np.unique(depth_crust_select, return_index=True)
    depth_crust_select = unique_depths
    vs_crust_select = vs_crust_select[unique_indices]
    
    # mantle
    vs_mantle_select[:-1] = vs_mantle_select[:-1] + np.random.uniform(vs_perturb_range[0], vs_perturb_range[1], len(vs_mantle_select)-1)
    # Smooth mantle nodes
    if len(vs_mantle_select) >= 2:
        vs_mantle_select = gaussian_filter1d(vs_mantle_select, sigma=gaussian_smooth_sigma)
    # not change the last node
    vs_mantle_select[-1] = vs[-1]
    # remove the same depth nodes
    unique_depths, unique_indices = np.unique(depth_mantle_select, return_index=True)
    depth_mantle_select = unique_depths
    vs_mantle_select = vs_mantle_select[unique_indices]

    #----------------------------------------------------
    #  interpolate the velocity
    #----------------------------------------------------
    # Connect crust and mantle nodes for spline interpolation
    controle_node_depth = np.concatenate([depth_crust_select,[depth[moho_idx]],depth_mantle_select])
    controle_node_vs = np.concatenate([vs_crust_select,[new_moho_vs],vs_mantle_select])
    
    # interpolate the velocity
    spl_crust_new = PchipInterpolator(controle_node_depth, controle_node_vs)
    vs_perturbed = spl_crust_new(depth)
    
    if return_nodes:
        return vs_perturbed, controle_node_depth, controle_node_vs
    else:
        return vs_perturbed

# -------------------------------------------------------
#  remove the sandwich layers
# -------------------------------------------------------
def combine_thin_sandwich(vs, depth, thickness_threshold=10, uniform_thickness=1, gradient_threshold=0.005, return_idx=False):
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
    new_depth = depth.copy()
    
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
def smooth_vs_by_node_interp(vs, depth, n_nodes=10,method="spline"):
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

###################################################################################################
# Note!
# The following functions are used to show the process of datasets construction
# Though they can be used to construct the datasets, while not adaptive to any types of data source
# We encourage users to use the functions in the SWIDP toolkit directly
# when developing customized workflows.
###################################################################################################

import os
def load_Central_and_Western_US_velocity_model(data_path):
    """load the velocity model
    """
    files = os.listdir(data_path)
    # read the data
    loc,vs_temp,crustal_thick = [],[],[]
    max_len = 0
    for file in files:
        lines = open(os.path.join(data_path, file))
        temp_data = []
        for i,line in enumerate(lines):
            line = line.strip().split()
            line = list(map(float,line))
            if i == 0:
                loc.append([line[0],line[1]])
                crustal_thick.append(line[2])
            else:
                temp_data.append([line[0],line[1]])
        vs_temp.append(temp_data)
        max_len = np.max([max_len,len(temp_data)])
    vs = np.ones(((len(vs_temp),max_len,2)))*-1
    for i,data in enumerate(vs_temp):
        vs[i,:len(data),:] = data
    loc = np.array(loc)
    vs = np.array(vs)
    crustal_thick = np.array(crustal_thick)
    depth = vs[:,:,0]
    vs = vs[:,:,1]
    return depth,vs

def extract_velocity_profiles(data_path,data_name = "Central_and_Western_US"):
    if data_name == "Central_and_Western_US":
        return load_Central_and_Western_US_velocity_model(data_path)
    else:
        raise ValueError(f"Invalid data name: {data_name}")

def interpolate_velocity_profiles(depth,vs,depth_interp):
    """interpolate the velocity profiles
    """
    from scipy.interpolate import interp1d
    max_depth = 300+0.1
    depth_interp = np.concatenate([
        np.arange(0, max_depth, 1)
    ])
    vs_interp = np.zeros((vs.shape[0],depth_interp.shape[0]))
    for i in range(vs.shape[0]):
        vs_temp = vs[i,:]
        depth_temp = depth[i,:]
        depth_interp_temp = depth_interp[:np.argmin(np.abs(depth_interp-depth_temp.max()))]
        f = interp1d(depth_temp.reshape(-1),vs_temp,kind='nearest')
        vs_temp = f(depth_interp_temp)
        vs_interp[i,:len(vs_temp)] = vs_temp
    vs_interp[:,len(vs_temp):] = vs_interp[:,len(vs_temp)-1].reshape(-1,1)
    return depth_interp,vs_interp

from p_tqdm import p_map
class SWIModel:
    def __init__(self,data_path,data_name = "Central_and_Western_US"):
        self.data_path = data_path
        self.data_name = data_name

    def extract_velocity_profiles(self):
        return extract_velocity_profiles(self.data_path,self.data_name)
    
    def interpolate_velocity_profiles(self,depth,vs,depth_interp):
        return interpolate_velocity_profiles(depth,vs,depth_interp)
    
    def combine_thin_sandwich(self,vs,depth,thickness_threshold=10,uniform_thickness=1,gradient_threshold=0.005,return_idx=False):
        vs = p_map(combine_thin_sandwich,
                   vs,
                   list(depth.reshape(1,-1))*len(vs),
                   [thickness_threshold]*len(vs),
                   [uniform_thickness]*len(vs),
                   [gradient_threshold]*len(vs),
                   [return_idx]*len(vs))
        vs = np.array(vs)
        depth = np.array(depth)
        return depth,vs
    
    def smooth_vs_by_node_interp(self,vs,depth,n_nodes=10,method="spline"):
        vs = p_map(smooth_vs_by_node_interp,
                   vs,
                   list(depth.reshape(1,-1))*len(vs),
                   [n_nodes]*len(vs),
                   [method]*len(vs))
        vs = np.array(vs)
        depth = np.array(depth)
        return depth,vs
    
    def find_moho_depth(self,vs,depth,moho_depth_range=[5,90],gradient_search=True,gradient_threshold=0.1,gradient_search_range=10):
        moho_idx = p_map(find_moho_depth,
                         vs,
                         list(depth.reshape(1,-1))*len(vs),
                         [moho_depth_range]*len(vs),
                         [gradient_search]*len(vs),
                         [gradient_threshold]*len(vs),
                         [gradient_search_range]*len(vs))
        moho_idx = np.array(moho_idx)
        return moho_idx
    
    def augment_crust_moho_mantle(self,vs,depth,moho_idx,vs_perturb_range=[-0.1,0.1],crust_nodes_range=[3,10],mantle_nodes_range=[6,12],moho_shift_range=10,gaussian_smooth_sigma=2,return_nodes=False,random_seed=None):
        vs_augmented = p_map(augment_crust_moho_mantle,
                             vs,
                             list(depth.reshape(1,-1))*len(vs),
                             moho_idx,
                             [vs_perturb_range]*len(vs),
                             [crust_nodes_range]*len(vs),
                             [mantle_nodes_range]*len(vs),
                             [moho_shift_range]*len(vs),
                             [gaussian_smooth_sigma]*len(vs),
                             [return_nodes]*len(vs),
                             [random_seed]*len(vs))
        depth = np.array(depth)
        vs_augmented = np.array(vs_augmented)
        return depth,vs_augmented
    
    def transform_vs_to_vel_model(self,vs,depth):
        vel_model = p_map(transform_vs_to_vel_model,
                         vs,
                         list(depth.reshape(1,-1))*len(vs))
        vel_model = np.array(vel_model)
        return vel_model
    
    def generate_mixed_samples(self,num_samples=100,start=0.2,end=10,uniform_num=50,log_num=20,random_num=30):
        t = generate_mixed_samples(num_samples,start,end,uniform_num,log_num,random_num)
        return t
    
    def calculate_dispersion(self,vel_model,t):
        t = np.ones((len(vel_model),len(t)))*t
        disp = p_map(calculate_dispersion,
                     vel_model,
                     list(t))
        disp = np.array(disp)
        return disp
    
    def save_velocity_model(self,save_path,vel_model):
        return np.savez(save_path,vel_model)
    
    def save_dispersion_curves(self,save_path,dispersion_curves):
        return np.savez(save_path,dispersion_curves)