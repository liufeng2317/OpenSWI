import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import interpolate
from concurrent.futures import ThreadPoolExecutor
from .data_augmentation import *
from typing import List


def train_collate_fn(batch):
    """
    Collate function for batching data during training. This function takes a list of samples (tuples of data, masks, 
    labels, and labels_used_layer), processes them to ensure uniformity in sequence length, applies padding, 
    and prepares the data for feeding into a neural network model.

    Parameters:
    - batch (list of tuples)                : A list where each element is a tuple containing:
        - data (torch.Tensor)               : The input data for the model, typically of shape (batch_size, seq_length).
        - data_mask (torch.Tensor)          : A mask for the input data, indicating valid (1) and padded (0) values.
        - labels (torch.Tensor)             : The target labels corresponding to the input data.
        - labels_used_layer (torch.Tensor)  : A tensor indicating which layers were used for the labels.

    Returns:
    - padded_data (torch.Tensor): The input data padded to the maximum sequence length within the batch, 
        with padding values set to -1 for consistency.
    - padded_data_mask (torch.Tensor): A mask for the padded input data, marking positions that are padded.
    - labels (torch.Tensor): A stacked tensor of labels for the entire batch.
    - labels_uselayer (torch.Tensor): A stacked tensor of labels_used_layer for the entire batch.

    This function performs the following:
        1. Pads the input data sequences to the maximum length within the batch.
        2. Adjusts any input values that are zero to -1 (for consistency in handling padding).
        3. Creates a mask to indicate padded positions in the input data.
        4. Stacks the labels and labels_used_layer tensors into single tensors for easy batch processing.
    """
    data, data_mask, labels,labels_used_layer = zip(*batch)
    # Find the maximum length in the batch
    max_length = max(d.size(1) for d in data)
    
    # Pad the data to the maximum length
    padded_data = [torch.cat([d, torch.ones(d.size(0), max_length - d.size(1))*(-1)], dim=1) if d.size(1) < max_length else d for d in data]
    padded_data = torch.stack(padded_data, dim=0)
    
    # change the zero input to -1
    mask = padded_data <= 0
    padded_data[mask] = -1
    
    # Labels are already tensors, no need to convert
    labels = torch.stack(labels)  # Stack labels into a single tensor
    
    # Labels do not need padding
    padded_data_mask = (padded_data[:,1, :] == -1) & (padded_data[:,2, :] == -1)
    
    # labels used 
    labels_uselayer = torch.stack(labels_used_layer)
    return padded_data,padded_data_mask,labels,labels_uselayer

def test_collate_fn(batch):

    data, data_mask = zip(*batch)
    # Find the maximum length in the batch
    max_length = max(d.size(1) for d in data)
    
    # Pad the data to the maximum length
    padded_data = [torch.cat([d, torch.ones(d.size(0), max_length - d.size(1))*(-1)], dim=1) if d.size(1) < max_length else d for d in data]
    padded_data = torch.stack(padded_data, dim=0)
    
    # change the zero input to -1
    mask = padded_data <= 0
    padded_data[mask] = -1
    
    # Labels do not need padding
    padded_data_mask = (padded_data[:,1, :] == -1) & (padded_data[:,2, :] == -1)
    
    return padded_data,padded_data_mask

def auto_collate_fn(batch):
    """
    Automatically determines whether to use train or test collate function based on the batch structure.
    
    Parameters:
    - batch (list of tuples): A list of tuples containing either training or testing data
    
    Returns:
    - The appropriate collated data based on whether it's training or testing data
    """
    # Check if batch contains training data (4 elements) or testing data (2 elements)
    is_training = len(batch[0]) == 4
    
    if is_training:
        return train_collate_fn(batch)
    else:
        return test_collate_fn(batch)

class DispersionDatasets(Dataset):
    """ used in pre-training
    This class represents a dataset for DispFormer, suitable for training or evaluating a model
    that predicts subsurface velocity models.The dataset supports both training and evaluation modes, 
    with options to augment the data, add noise, and mask parts of the input sequences to improve model robustness and generalization.

    Attributes:
    - input_data_path (str): Path to the input dispersion data file, which contains three columns per sample: 
      [period, phase velocity, group velocity].
    - input_label_path (str): Path to the velocity model file, which contains four columns per sample: 
      [depth, P-wave velocity (vp), S-wave velocity (vs), density (rho)].
    - train (bool): A flag indicating whether the dataset is used for training or testing.
    - interp_layer (bool): Whether to automatically interpolate the layers to have equal thickness.
        - layer_thickness (float): The thickness of each interpolated layer (used when `interp_layer=True`).
        - layer_number (int): The number of layers in the velocity model, used for interpolation and layer extraction.
        - layer_interp_kind (str): The interpolation method for adjusting the layer thickness, e.g., 'nearest', 'linear'.
    - layer_used_range (List[float]): The range of layers (depth range) to use from the velocity model. (not used in pre-training)
    - num_workers (int): The number of workers for parallel data loading (used when loading large datasets).
    - augmentation_train_data (bool): Whether to apply data augmentation during training (such as noise, masking, etc.).
        - noise_level (float): The standard deviation of the noise added to the dispersion data during training (for augmentation).
        - mask_ratio (float): The fraction of the input data to randomly mask during training as part of the data augmentation.
        - remove_phase_ratio (float): The fraction of phase velocity data to randomly remove for augmentation during training.
        - remove_group_ratio (float): The fraction of group velocity data to randomly remove for augmentation during training.
        - max_masking_length (int): The maximum length of sequences to mask in the dispersion data during training.

    Methods:
    - __len__: Returns the total number of samples in the dataset.
    - __getitem__: Loads and returns a sample from the dataset, including both input data and corresponding labels.
    - _load_input_data: Loads the dispersion data from the input file (not implemented here, can be customized).
    - _load_output_data: Loads the velocity model data from the label file (not implemented here, can be customized).
    - _interp_vs: Interpolates the layers to a uniform thickness based on the specified settings.
    - augmentation: Optionally applies augmentation techniques (e.g., noise, masking) to the training data.

    Usage:
    The `DispersionDatasets` class is designed to be used with PyTorch's `DataLoader` for efficient data loading 
    and batching during training and evaluation. It can handle both the dispersion data (input) and the velocity 
    model labels (output), along with optional data augmentation and masking for training.

    Example:
        ```python
        dataset = DispersionDatasets(input_data_path="path_to_dispersion_data.csv", 
                                    input_label_path="path_to_velocity_model.csv", 
                                    train=True, 
                                    augmentation_train_data=True)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        ```
    """

    def __init__(self, 
                 input_data_path: str           = "", 
                 input_label_path: str          = "", 
                 input_data_period_range: List[float] = [-1,-1],
                 train: bool                    = True,
                 interp_layer: bool             = False,
                 interp_layer_thickness: float  = 0.5,
                 interp_layer_number: int       = 100,
                 interp_kind: str               = "nearest",
                 layer_used_range: List[float]  = [0, 0],
                 num_workers: int               = 0,
                 normalize_input_data: bool     = False,
                 aug_train_data: bool           = False,
                 aug_noise_level: float         = 0,
                 aug_random_mask_ratio: float   = 0,
                 aug_max_masking_length: int    = 0,
                 aug_remove_phase_ratio: float  = 0,
                 aug_remove_group_ratio: float  = 0,
                 aug_varylength: bool           = False,
                 aug_varylength_start_range: List[int] = [0,5],
                 aug_varylength_min_length: int = 30,
                 aug_varylength_c1: float       = 1/3,
                 aug_varylength_c2: float       = 1/2,
                 aug_varylength_c3: float       = 1.1,
        ):
        self.input_data_path        = input_data_path
        self.input_label_path       = input_label_path
        self.interp_layer           = interp_layer
        self.interp_layer_thickness = interp_layer_thickness
        self.interp_layer_number    = interp_layer_number
        self.interp_kind            = interp_kind
        self.layer_used_start       = layer_used_range[0]
        self.layer_used_end         = layer_used_range[1]
        self.num_workers            = num_workers
        self.normalize_input_data   = normalize_input_data
        self.input_data_period_range_low = input_data_period_range[0]
        self.input_data_period_range_high = input_data_period_range[1]
        
        # Augmentation parameters
        self.aug_train_data             = aug_train_data
        self.aug_noise_level            = aug_noise_level
        self.aug_random_mask_ratio      = aug_random_mask_ratio
        self.aug_max_masking_length     = aug_max_masking_length
        self.aug_remove_phase_ratio     = aug_remove_phase_ratio
        self.aug_remove_group_ratio     = aug_remove_group_ratio
        
        self.aug_varylength             = aug_varylength
        self.aug_varylength_start_range = aug_varylength_start_range
        self.aug_varylength_min_length  = aug_varylength_min_length
        self.aug_varylength_c1          = aug_varylength_c1
        self.aug_varylength_c2          = aug_varylength_c2
        self.aug_varylength_c3          = aug_varylength_c3
        
        # Load and process input dataset
        self.input_dataset = self._load_input_data(input_data_path)
        self.input_masks = (self.input_dataset[:, 1, :] <= 0) & (self.input_dataset[:, 2, :] <= 0)

        # Load and process output dataset
        self.train = train
        if train:
            self.output_dataset = self._load_output_data(input_label_path, interp_layer, num_workers)
            self.layer_used_end = self.output_dataset.shape[-1] if self.layer_used_end == 0 else self.layer_used_end
            self.depth = self.output_dataset[0, 0, :]
            
    def _load_input_data(self, path: str) -> torch.Tensor:
        """Load input dataset from the specified path."""
        try:
            input_dataset = np.load(path)["data"].transpose(0, 2, 1)
            # transform the value <=0 to -1
            input_dataset[input_dataset <= 0] = -1
            return torch.tensor(input_dataset, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Error loading input data from {path}: {e}")

    def _load_output_data(self, path: str, interp_layer: bool, num_workers: int) -> torch.Tensor:
        """Load output dataset from the specified path and optionally interpolate."""
        try:
            output_dataset = np.load(path)["data"].transpose(0, 2, 1) 
            if output_dataset.shape[1] == 4:
                output_dataset = output_dataset[:, [0,2], :] # get the thickness and vs
            if interp_layer:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    output_dataset = list(executor.map(self._interp_vs, output_dataset))
                output_dataset = np.array(output_dataset)
            return torch.tensor(output_dataset, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Error loading output data from {path}: {e}")

    def _interp_vs(self, output_data):
        """Interpolate 1D velocity model."""
        depth, vs = output_data[0, :], output_data[1, :]
    
        # Create an interpolation function
        F = interpolate.interp1d(depth, vs, kind=self.interp_kind, fill_value="extrapolate")
        
        # Generate interpolated depth points
        interp_depth = np.arange(depth.min(), depth.max(), self.interp_layer_thickness)
        interp_vs = F(interp_depth)
        
        # Check the length of the interpolated velocity data
        num_interp_points = len(interp_vs)

        if num_interp_points < self.interp_layer_number:
            # Calculate the number of elements to add
            num_to_add = self.interp_layer_number - num_interp_points
            
            # Calculate the additional depths and vs values
            additional_depths = interp_depth[-1] + self.interp_layer_thickness * np.arange(1, num_to_add + 1)
            additional_vs = np.full(num_to_add, vs[-1])  # Use the last vs value for padding
            
            # Concatenate the additional values
            interp_depth = np.concatenate((interp_depth, additional_depths))
            interp_vs = np.concatenate((interp_vs, additional_vs))
        else:
            # Trim the interpolated arrays to the desired layer number
            interp_depth = interp_depth[:self.interp_layer_number]
            interp_vs = interp_vs[:self.interp_layer_number]

        return np.vstack((interp_depth, interp_vs))
    
    def augmentation(self, input_data):
        """Applies data augmentation methods to the input data."""
        # Early exit if no augmentation is needed
        if self.aug_noise_level <= 0 and self.aug_random_mask_ratio <= 0 and self.aug_remove_group_ratio <= 0 and self.aug_remove_phase_ratio <= 0:
            return input_data

        # Add Gaussian noise if noise_level > 0
        if self.aug_noise_level > 0:
            input_data = add_gaussian_noise(input_data, noise_level=self.aug_noise_level)
        
        # Apply random masking if mask_ratio > 0
        if self.aug_random_mask_ratio > 0:
            input_data = random_masking(input_data, mask_ratio=self.aug_random_mask_ratio)
        
        # Randomly remove phase or group velocity if either ratio > 0
        if self.aug_remove_group_ratio > 0 or self.aug_remove_phase_ratio > 0:
            input_data = random_remove_phase_or_group(
                input_data, 
                remove_phase_ratio=self.aug_remove_phase_ratio, 
                remove_group_ratio=self.aug_remove_group_ratio, 
                masking_value=-1
            )
        return input_data

    def _normalize_input_data(self, input_data):
        """
        Normalize the input data by subtracting mean and dividing by standard deviation.
        Only normalizes non-masked values (>0).
        
        Parameters:
            input_data (torch.Tensor): Input tensor of shape (3, seq_length) containing 
                                    [period, phase velocity, group velocity]
        Returns:
            torch.Tensor: Normalized input data
        """
        # Only normalize non-masked values (>0)
        phase_mask  = input_data[1, :] > 0  
        group_mask  = input_data[2, :] > 0
        
        if phase_mask.any():
            phase_mean = input_data[1, phase_mask].mean()
            phase_std  = input_data[1, phase_mask].std()
            input_data[1, phase_mask] = (input_data[1, phase_mask] - phase_mean) / phase_std

        if group_mask.any():
            group_mean = input_data[2, group_mask].mean()
            group_std = input_data[2, group_mask].std()
            input_data[2, group_mask] = (input_data[2, group_mask] - group_mean) / group_std

        return input_data

    def vary_length(self, input_data, masking_value=-1):
        """
        Randomly selects a region of the input data to simulate varying periods and lengths by masking
        parts of the data. This function helps simulate varying data lengths, which can be used for 
        data augmentation, ensuring that the model can generalize well to input data of different lengths.

        Parameters:
        -----------
        input_data : torch.Tensor or np.ndarray
            The input data that will undergo masking. It should have the shape of (batch_size, num_points).
        masking_value : int, optional
            The value to use for masking the data. By default, the masking value is -1. This will replace
            the values outside the valid data range.
        min_data_length : int, optional
            The minimum length of valid data (i.e., the region that remains unmasked). The default is 30.
        min_end_idx : int, optional
            The minimum index for the end of the valid data region. This ensures that even with random masking,
            the valid data region will not be too short. Default is 60.
            
        Returns:
        --------
        torch.Tensor or np.ndarray
            The input data with certain regions masked according to the specified rules. This is the same 
            shape as the input data, with parts of it replaced by the `masking_value`.

        Process Overview:
        -----------------
        1. The function randomly selects a starting index (`mask_begin_idx`) for the valid region.
        2. A random length for the valid region is chosen, ensuring the region is at least `min_data_length`.
        3. The end index of the valid data region (`mask_end_idx`) is adjusted to ensure it meets the minimum 
        valid length and does not exceed the data size.
        4. The data outside the valid region is masked by assigning the `masking_value` to the corresponding elements.
        5. The output is the input data with masked regions.

        Example:
        --------
        input_data = torch.ones((1, 200))  # (batch_size=1, num_points=200)
        result = vary_length(input_data)
        print(result)  # The data will have masked regions, with values outside the valid region replaced by -1.
        """
        
        # Randomly choose the starting index and length for the valid data
        num_points = input_data.shape[1]
        
        if self.aug_varylength_start_range[0] == -1 and self.aug_varylength_start_range[1] == -1:
            mask_begin_idx = np.random.randint(0, 5)
        else:
            mask_begin_idx = np.random.randint(self.aug_varylength_start_range[0], self.aug_varylength_start_range[1])
        
        # before
        # mask_end_idx = mask_begin_idx + np.random.randint(self.aug_varylength_min_length, max(self.aug_varylength_min_length + 1, num_points - self.aug_varylength_min_length))
        # mask_length = np.random.randint(30, max(30 + 1, num_points - 30)) # 30 -> 270
        # mask_end_idx = mask_begin_idx + mask_length
        # mask_end_idx = max(105, mask_end_idx) # begin_idx - > [105:295]
        
        # after
        mask_length = np.random.randint(self.aug_varylength_min_length, max(self.aug_varylength_min_length + 1, num_points - mask_begin_idx))
        mask_end_idx = mask_begin_idx + mask_length
        
        # Ensure mask_end_idx does not exceed the length of input_data
        mask_end_idx = min(mask_end_idx, num_points)

        # Apply masking to regions outside the valid range
        # ensure at least one point is not masked
        if input_data[1, mask_begin_idx:mask_end_idx].sum() <= 0 and input_data[2, mask_begin_idx:mask_end_idx].sum() <= 0:
            pass
        else:
            input_data[1:, :mask_begin_idx] = masking_value  # Mask the beginning
            input_data[1:, mask_end_idx:]   = masking_value  # Mask the end
        
        return input_data
    
    def _aug_varylength(self, input_data):
        """
            Calculate the dynamic range of the inversion depth based on the input data.
        """
        input_data = self.vary_length(input_data)
        
        phase_mask = input_data[1, :] > 0
        group_mask = input_data[2, :] > 0

        # calcualte the dynamic range of the inversion depth
        phase_period    = input_data[0, phase_mask]
        phase_velocity  = input_data[1, phase_mask] 
        if phase_mask.any():
            phase_min_period,phase_max_period = phase_period.min(),phase_period.max()
            min_depth_phase = self.aug_varylength_c1 * (phase_min_period * phase_velocity[phase_period.argmin()])
            max_depth_phase = self.aug_varylength_c3 * (phase_max_period * phase_velocity[phase_period.argmax()])
        else:
            min_depth_phase, max_depth_phase = None, None

        # Group depth calculation
        group_period    = input_data[0, group_mask]
        group_velocity  = input_data[2, group_mask]
        if group_mask.any():
            group_min_period,group_max_period = group_period.min(),group_period.max()
            min_depth_group = self.aug_varylength_c2 * (group_min_period * group_velocity[group_period.argmin()])
            max_depth_group = self.aug_varylength_c3 * (group_max_period * group_velocity[group_period.argmax()])
        else:
            min_depth_group, max_depth_group = None, None

        # Calculate min and max depth indices
        min_depth = min(filter(None, [min_depth_phase, min_depth_group]), default=None)
        max_depth = max(filter(None, [max_depth_phase, max_depth_group]), default=None)

        min_depth_idx = max(0, np.argmin(np.abs(self.depth - min_depth))-1) if min_depth is not None else 0
        max_depth_idx = min(len(self.depth), np.argmin(np.abs(self.depth - max_depth))+1) if max_depth is not None else len(self.depth)

        return input_data, min_depth_idx, max_depth_idx

    def __getitem__(self, index):
        input_data = self.input_dataset[index].clone() # (3, seq_length)
        if self.input_data_period_range_low != -1 or self.input_data_period_range_high != -1:
            mask_low = (input_data[0, :] >= self.input_data_period_range_low) if self.input_data_period_range_low != -1 else True
            mask_high = (input_data[0, :] <= self.input_data_period_range_high) if self.input_data_period_range_high != -1 else True
            mask = mask_low & mask_high
            input_data = input_data[:, mask]
        
        # data augmentation
        if self.aug_train_data and self.train:
            input_data = self.augmentation(input_data)

        # normalize the input data
        if self.normalize_input_data:
            input_data = self._normalize_input_data(input_data)
            
        # calculate the varying length of input data & recalculate the masks
        if self.aug_varylength: 
            input_data, min_depth_idx, max_depth_idx = self._aug_varylength(input_data)
        else:
            min_depth_idx, max_depth_idx = 0, len(self.depth)
            
        input_mask = (input_data[1, :] <= 0) & (input_data[2, :] <= 0)
        
        if self.train:
            output_data = self.output_dataset[index]
            used_layer = torch.tensor([min_depth_idx, max_depth_idx], dtype=torch.int)
            return input_data, input_mask, output_data, used_layer
        else:
            return input_data, input_mask

    def __len__(self):
        return len(self.input_dataset)