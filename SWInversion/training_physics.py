import sys
sys.path.append("..")
sys.path.append("../ADsurf")
sys.path.append("../ADsurf/_cps")
from ADsurf._ADsurf import *
from SWInversion.dispersion import *
import ADsurf._cps._surf96_vector_gpu as surf_vector_iter_gpu
import ADsurf._cps._surf96_vectorAll_gpu as surf_vector_all_gpu
import time
import torch
import numpy as np
from SWInversion.model.dispformer import DispersionTransformer as DispFormer
from SWInversion.model.dispformer_local_global import DispersionTransformer as DispFormer_local_global
from SWInversion.model.sfnet import S2vpNet as SfNet
from SWInversion.model.FCNN import FCNN
from SWInversion.model.Unet import UNet1D
from SWInversion.misfits import NMSE, MAPE,MSE,MAE
from SWInversion.args import get_save_path_name
import os


class EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

###############################################################
#  Forward modeling using ADsurf
###############################################################

def check_ADsurf_inputs(thick,vp,vs,rho,tlist,vlist):
    thick = thick.clamp(min=0.001)
    vs = vs.clamp(min=0.1,max=4.6)
    vp = vp.clamp(min=0.1)
    rho = rho.clamp(min=0.1)
    return thick,vp,vs,rho,tlist,vlist

def ADsurf_forward(thick,vs,tlist,vlist,adsurf_normalize=False):
    """
    vs is a tensor of shape (batch_size, n_layer)
    tlist is a tensor of shape (batch_size, n_layer)
    vlist is a tensor of shape (batch_size, n_layer)
    """
    mask = (tlist > 0.0) * (vlist > 0.0)
    tlist[~mask] = 0.5
    vlist[~mask] = 3
    
    device = vs.device
    # transform the S-wave velocity to velocity model (Brocher, 2005)
    with torch.no_grad():
        vp       = transform_vs_to_vp(vs)
        rho      = transform_vp_to_rho(vp)
    thick,vp,vs,rho,tlist,vlist = check_ADsurf_inputs(thick,vp,vs,rho,tlist,vlist)
    wave = "rayleigh"
    algorithm = "dunkin"
    ifunc = ifunc_list[algorithm][wave]
    llw = 0 if vs[0][0] <= 0.0 else -1
    F = surf_vector_all_gpu.dltar_vector(vlist, tlist, thick, vp, vs, rho, ifunc, llw, device=device)    
    
    if adsurf_normalize:
        clist = torch.arange(vlist.min(),vlist.max()+0.01,0.01,device=device)
        Clist = torch.ones((tlist.shape[0],clist.shape[0]),device=device)*clist
        with torch.no_grad():
            det  = surf_matrix_all_gpu.dltar_matrix(Clist, tlist, thick, vp, vs, rho, ifunc, llw, device=device)
        # remove nan values in det
        det = torch.where(torch.isnan(det), torch.zeros_like(det), det)
        F = F[mask]/(torch.max(det,dim=1).values - torch.min(det,dim=1).values)[mask]
        F = (1e-1)**torch.abs(F) - 1
    else:
        F = F[mask]
    return F

#----------------------------------------------------
#  Training loss function
#----------------------------------------------------
from SWInversion.misfits import NMSE, MAPE,MSE,MAE
def get_loss_fn(loss_type="NMSE"):
    if loss_type.lower() == "nmse":
        loss_fn = NMSE
    elif loss_type.lower() == "mape":
        loss_fn = MAPE
    elif loss_type.lower() == "mse":
        loss_fn = MSE
    elif loss_type.lower() == "mae":
        loss_fn = MAE
    else:
        raise ValueError(f"Loss type {loss_type} not found")
    return loss_fn


def physical_loss_fn(thick,vs,tlist,vlist,physical_weights,adsurf_normalize=False):
    """
    Calculate the physical loss of the model
    """
    # try:
    physical_output = ADsurf_forward(thick,vs,tlist,vlist,adsurf_normalize=adsurf_normalize) # [batch_size, loss]
    # Check if there are any NaN values and handle them
    nan_mask = torch.isnan(physical_output)
    if nan_mask.any():
        print(f"Warning: Found {nan_mask.sum().item()} NaN values in physical output")
        physical_output = physical_output[~nan_mask]
        if physical_output.numel() == 0:
            print("Warning: All physical outputs were NaN, returning zero tensor")
            return torch.tensor(0.0, device=physical_output.device)
    # except:
    #     physical_output = torch.tensor(0.0, device=vs.device)

    # physical loss
    physical_loss = torch.sum(torch.abs(physical_output))
            
    return physical_loss*physical_weights

#############################################################################
# Model Training/Validation/Testing
#############################################################################

# update the model and the physical loss at the same time
def train_model(model,train_loader,optimizer,scheduler,loss_fn,reg_fn,
                cal_phy_loss=True,
                cal_phy_loss_batch_interval=1, # update the physical loss every cal_phy_loss_batch_interval batches
                phy_loss_weight=1e-4,
                device="cpu",
                model_name="DispFormer"):
    model.train()
    # model loss
    train_data_loss = 0.0
    train_phy_loss = 0.0
    for batch_idx, (input_data, input_mask, target_data, used_layer) in enumerate(train_loader):        
        # input data
        tlist = input_data[:,0,:].to(device).clone()
        vlist = input_data[:,1,:].to(device).clone()
        if model_name.lower() == "dispformer" or model_name.lower() == "dispformer_local_global" or model_name.lower() == "unet":
            # period, phase velocity, group velocity
            input_data  = input_data.to(device)
            input_mask  = input_mask.to(device)
            target_data = target_data.to(device)
        elif model_name.lower() == "sfnet":
            # phase velocity, group velocity
            input_data  = input_data[:,1:,:].to(device)
            target_data = target_data.to(device)
        elif model_name.lower() == "fcnn":
            # period, phase velocity
            input_data  = input_data[:,0:2,:].to(device)
            target_data = target_data.to(device)
        
        # model forward
        optimizer.zero_grad()
        if model_name.lower() == "dispformer" or model_name.lower() == "dispformer_local_global":
            output = model(input_data, input_mask)
        elif model_name.lower() == "sfnet" or model_name.lower() == "fcnn" or model_name.lower() == "unet":
            output = model(input_data)
            
        # model misfit calculation
        model_loss = torch.tensor(0.0,device=device,dtype=torch.float32)
        for i in range(used_layer.shape[0]):
            model_loss = model_loss + loss_fn(output[i][used_layer[i,0]:used_layer[i,1]], target_data[i,1,used_layer[i,0]:used_layer[i,1]])
            if reg_fn is not None:
                reg_loss = reg_fn(output[i][used_layer[i,0]:used_layer[i,1]])
                model_loss += reg_loss
        model_loss = model_loss / used_layer.shape[0]

        # physical loss calculation
        physical_loss = torch.tensor(0.0,device=device,dtype=torch.float32)
        if cal_phy_loss and batch_idx % cal_phy_loss_batch_interval == 0:
            vs = output.clone()
            depth = target_data[:,0,:]
            thick = torch.diff(depth)
            thick = torch.cat((thick,thick[:,-1].unsqueeze(1)),dim=1)
            max_used_layer = torch.max(used_layer[:,1])
            thick = thick[:,:max_used_layer]
            vs = vs[:,:max_used_layer]
            physical_loss = physical_loss_fn(thick,vs,tlist,vlist,phy_loss_weight,adsurf_normalize=True)
            
        loss = model_loss + physical_loss
        loss.backward()
        optimizer.step()
        
        train_data_loss += model_loss.item()
        train_phy_loss += physical_loss.item()
        
    # update the learning rate
    scheduler.step()
    return train_data_loss/len(train_loader),train_phy_loss/len(train_loader)

def valid_model(model,val_loader,loss_fn,device="cpu",model_name="DispFormer"):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (input_data, input_mask, target_data, used_layer) in enumerate(val_loader):
            # input data
            if model_name.lower() == "dispformer" or model_name.lower() == "dispformer_local_global" or model_name.lower() == "unet":
                input_data = input_data.to(device)
                input_mask = input_mask.to(device)
                target_data = target_data.to(device)
            elif model_name.lower() == "sfnet":
                input_data = input_data[:,1:,:].to(device)
                target_data = target_data.to(device)
            elif model_name.lower() == "fcnn":
                input_data = input_data[:,0:2,:].to(device)
                target_data = target_data.to(device)
            
            # model forward
            if model_name.lower() == "dispformer" or model_name.lower() == "dispformer_local_global":
                output = model(input_data, input_mask)
            elif model_name.lower() == "sfnet" or model_name.lower() == "fcnn" or model_name.lower() == "unet":
                output = model(input_data)
            
            # misfit calculation
            loss = 0.0
            for i in range(used_layer.shape[0]):
                loss += loss_fn(output[i][used_layer[i,0]:used_layer[i,1]], target_data[i,1,used_layer[i,0]:used_layer[i,1]])
            loss = loss / used_layer.shape[0]
            val_loss += loss.item()
    return val_loss/len(val_loader)

def test_model(model,test_loader,loss_fn,device="cpu",model_name="DispFormer"):
    model.eval()
    test_loss = 0.0
    test_targets,test_outputs,test_used_layer = [],[],[]
    with torch.no_grad():
        for batch_idx, (input_data, input_mask, target_data, used_layer) in enumerate(test_loader):
            # input data
            if model_name.lower() == "dispformer" or model_name.lower() == "dispformer_local_global" or model_name.lower() == "unet":
                # period, phase velocity, group velocity
                input_data = input_data.to(device)  
                input_mask = input_mask.to(device)
                target_data = target_data.to(device)
            elif model_name.lower() == "sfnet":
                # phase velocity, group velocity
                input_data = input_data[:,1:,:].to(device)
                target_data = target_data.to(device)
            elif model_name.lower() == "fcnn":
                # period, phase velocity
                input_data = input_data[:,0:2,:].to(device)
                target_data = target_data.to(device)
            
            # model forward
            if model_name.lower() == "dispformer" or model_name.lower() == "dispformer_local_global":
                output = model(input_data, input_mask)
            elif model_name.lower() == "sfnet" or model_name.lower() == "fcnn" or model_name.lower() == "unet":
                output = model(input_data)
            
            # misfit calculation
            loss = 0.0
            for i in range(used_layer.shape[0]):
                loss += loss_fn(output[i][used_layer[i,0]:used_layer[i,1]], target_data[i,1,used_layer[i,0]:used_layer[i,1]])   
            loss = loss / used_layer.shape[0]
            test_loss += loss.item()
            test_targets.append(target_data[:,1,:].cpu().numpy())
            test_outputs.append(output.cpu().numpy())
            test_used_layer.append(used_layer.cpu().numpy())
    test_targets = np.concatenate(test_targets, axis=0)
    test_outputs = np.concatenate(test_outputs, axis=0)
    test_used_layer = np.concatenate(test_used_layer, axis=0)
    return test_loss/len(test_loader),test_targets,test_outputs,test_used_layer

################################################
# Training skills
################################################
from torch.optim.lr_scheduler import LambdaLR

def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """
    Return a learning rate scheduler with linear warmup and linear decay.
    
    Args:
        optimizer: torch optimizer.
        warmup_steps: number of steps to warm up.
        total_steps: total number of training steps.
    
    Returns:
        scheduler: a LambdaLR scheduler with warmup.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)


#----------------------------------------------------
#  Training Model
#----------------------------------------------------
def get_dl_model(args,load_trained_model=False,device=None):
    model_name      = args.model_name
    model_dim       = args.model_dim
    num_heads       = args.num_heads
    num_layers      = args.num_layers
    output_dim      = args.output_dim
    scale_factor    = args.scale_factor
    seq_length      = args.seq_length
    init_features   = args.init_features
    scale_factor    = args.scale_factor
    if device is None:
        device  = args.device
    if load_trained_model:
        save_path,save_path_name = get_save_path_name(args)
        saved_model_path = os.path.join(save_path,"best_model.pth")
    else:
        saved_model_path = None
    
    if model_name.lower() == "dispformer":
        model = DispFormer(model_dim=model_dim, 
                                    num_heads=num_heads, 
                                    num_layers=num_layers, 
                                    output_dim=output_dim, 
                                    scale_factor=scale_factor)
    elif model_name.lower() == "dispformer_local_global":
        model = DispFormer_local_global(model_dim=model_dim, 
                                    num_heads=num_heads, 
                                    num_layers=num_layers, 
                                    output_dim=output_dim, 
                                    scale_factor=scale_factor)
    elif model_name.lower() == "sfnet":
        model = SfNet(seq_length=seq_length, 
                      init_features=init_features, 
                      output_dim=output_dim, 
                      scale_factor=scale_factor)
    elif model_name.lower() == "fcnn":
        model = FCNN(seq_length=seq_length,
                     output_dim=output_dim,
                     scale_factor=scale_factor)
    elif model_name.lower() == "unet":
        model = UNet1D(
                      seq_length=seq_length,
                      in_channels=3,
                      out_channels=1,
                      init_features=init_features,
                      output_dim=output_dim,
                      scale_factor=scale_factor)
    model.to(device)
    if saved_model_path is not None:
        model.load_state_dict(torch.load(saved_model_path,map_location=device))
    return model