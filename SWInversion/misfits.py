import torch
import numpy as np

#################################################################
#                               Misfits  
#################################################################
def RMSE_np(output: np.ndarray, target: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between the output and target arrays.
    
    Args:
        output (np.ndarray): Predicted values of shape (N, M), with units typically in km/s.
        target (np.ndarray): Ground truth values of shape (N, M), same units as output.

    Returns:
        float: RMSE value, with the same unit as the input (e.g., km/s).
    """
    return np.sqrt(np.mean((output - target) ** 2))


def NMSE(output, target):
    """
    Computes the Normalized Mean Squared Error (NMSE) between the output and target tensors.
    
    Parameters:
    output (torch.Tensor): Predicted output.
    target (torch.Tensor): Ground truth target.
    
    Returns:
    torch.Tensor: The NMSE value.
    """
    return torch.mean(((output - target) / target) ** 2)
    # return torch.sum(((output - target) / target) ** 2)

def NMSE_np(output, target):
    """
    Computes the Normalized Mean Squared Error (NMSE) using NumPy arrays.
    
    Parameters:
    output (np.array): Predicted output.
    target (np.array): Ground truth target.
    
    Returns:
    float: The NMSE value.
    """
    return np.mean(((output - target) / target) ** 2)

def MAPE(output, target):
    """
    Computes the Mean Absolute Percentage Error (MAPE) between the output and target tensors.
    """
    return torch.mean(torch.abs((output - target) / target)) * 100

def MAPE_np(output, target):
    """
    Computes the Mean Absolute Percentage Error (MAPE) using NumPy arrays.
    
    Parameters:
    output (np.array): Predicted output.
    target (np.array): Ground truth target.
    
    Returns:
    float: The MAPE value, expressed as a percentage.
    """
    return np.mean(np.abs((output - target) / target)) * 100

def MSE(output, target):
    """
    Computes the Mean Squared Error (MSE) between the output and target tensors.
    """
    return torch.mean((output - target) ** 2)

def MSE_np(output, target):
    """
    Computes the Mean Squared Error (MSE) using NumPy arrays.
    
    Parameters:
    output (np.array): Predicted output.
    target (np.array): Ground truth target.
    
    Returns:
    float: The MSE value, scaled by 1000.
    """
    return np.mean((output - target) ** 2)

def MAE(output, target):
    """
    Computes the Mean Absolute Error (MAE) between the output and target tensors.
    """
    return torch.mean(torch.abs(output - target))

def MAE_np(output, target):
    """
    Computes the Mean Absolute Error (MAE) using NumPy arrays.
    
    Parameters:
    output (np.array): Predicted output.
    target (np.array): Ground truth target.
    """
    return np.mean(np.abs(output - target))

def MAE_layers_np(output, target):
    """
    Computes the Mean Absolute Error (MAE) for each layer (dimension) of the output using NumPy arrays.
    
    Parameters:
    output (np.array): Predicted output with multiple layers.
    target (np.array): Ground truth target with multiple layers.
    
    Returns:
    np.array: The MAE value for each layer, scaled by 100.
    """
    results = np.mean(np.abs(output - target), axis=0) * 100
    return results

def RMSE(output, target):
    """
    Computes the Root Mean Squared Error (RMSE) between the output and target tensors.
    """
    return torch.sqrt(MSE(output, target))

def RMSE_np(output, target):
    """
    Computes the Root Mean Squared Error (RMSE) using NumPy arrays.
    """
    return np.sqrt(MSE_np(output, target))  

#--------------------------------------
#  Metrics 
#--------------------------------------
def evaluate_metrics(output, target):
    mse  = MSE_np(output, target)
    mae  = MAE_np(output, target)
    mape = MAPE_np(output, target)
    nmse = NMSE_np(output, target)
    return mse, mae, mape, nmse

#----------------------------------------------------
#  Training loss function
#----------------------------------------------------
class LossFunction:
    def __init__(self, loss_type="NMSE"):
        self.loss_type = loss_type
        self.check_loss_type()
    
    def check_loss_type(self):
        if self.loss_type.lower() not in ["nmse", "mape", "mse", "mae"]:
            raise ValueError(f"Loss type {self.loss_type} not found")

    def __call__(self, output, target):
        if self.loss_type.lower() == "nmse":
            return NMSE(output, target)
        elif self.loss_type.lower() == "mape":
            return MAPE(output, target)
        elif self.loss_type.lower() == "mse":
            return MSE(output, target)
        elif self.loss_type.lower() == "mae":
            return MAE(output, target)

#################################################################
#               Velocity Regularization  
#################################################################
class VelocityRegularization:
    def __init__(self, reg_type="1order", reg_weight=1.0):
        self.reg_type = reg_type
        self.reg_weight = reg_weight
    
    def check_reg_type(self):
        if self.reg_type.lower() not in ["1order", "2order"]:
            raise ValueError(f"Regularization type {self.reg_type} not found")

    def __call__(self, vs):
        if vs.ndim == 1:
            vs = vs.unsqueeze(0)
        if self.reg_type == "1order":
            regu_term = torch.sum(torch.mean(torch.abs(vs[:,1:] - vs[:,:-1]), dim=0)) # (batch_size, n_layers-1) -> (batch_size) -> (1)
            return regu_term * self.reg_weight
        elif self.reg_type == "2order":
            regu_term = torch.sum(torch.mean(torch.abs(vs[:,2:] - 2*vs[:,1:-1] + vs[:,:-2]), dim=0)) # (batch_size, n_layers-2) -> (batch_size) -> (1)
            return regu_term * self.reg_weight
