import torch
import numpy as np
from SWInversion.model.dispformer import DispersionTransformer as DispFormer
from SWInversion.model.dispformer_local_global import DispersionTransformer as DispFormer_local_global
from SWInversion.model.dispformer_local_global_v1 import DispersionTransformer as DispFormer_local_global_v1
from SWInversion.model.sfnet import S2vpNet as SfNet
from SWInversion.model.FCNN import FCNN
from SWInversion.model.Unet import UNet1D
from SWInversion.misfits import NMSE, MAPE,MSE,MAE
from SWInversion.args import get_save_path_name
import os
import torch.nn.functional as F
from tqdm import tqdm

#----------------------------------------------------
# Early Stopping
#----------------------------------------------------
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

# ----------------------------------------------------
# Scheduler
# ----------------------------------------------------
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LinearLR, SequentialLR

def get_scheduler(optimizer, args, total_steps=None):
    scheduler_method = args.scheduler_method.lower()
    gamma = args.gamma
    
    # make sure the total_steps is not None
    if total_steps is None:
        total_steps = 1
    
    # calculate the warmup steps
    warmup_steps = int(args.warmup_ratio * total_steps) if hasattr(args, 'warmup_ratio') else 0
    
    if scheduler_method == "steplr+warmup":
        # convert the step_size from epoch unit to step unit
        # assume the step_size is in epoch unit
        steps_per_epoch = total_steps // args.num_epochs
        step_size_in_steps = max(1, args.step_size * steps_per_epoch)
        
        # warmup-scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-5,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # main-scheduler (StepLR)
        main_scheduler = StepLR(
            optimizer,
            step_size=step_size_in_steps,  # use the converted step_size
            gamma=gamma
        )
        
        # combine-scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]  # warmup-scheduler -> main-scheduler
        )
        
    elif scheduler_method == "cosine+warmup":
        # warmup-scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-5,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # cosine-scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,  # remaining steps
            eta_min=args.lr * 0.01  # final learning rate is 1% of the initial learning rate
        )
        
        # combine-scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
    elif scheduler_method == "steplr":
        # convert the step_size from epoch unit to step unit
        # steps_per_epoch = total_steps // args.num_epochs
        # step_size_in_steps = max(1, args.step_size * steps_per_epoch)
        
        scheduler = StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=gamma
        )
    
    else:
        # default-scheduler
        scheduler = StepLR(
            optimizer,
            step_size=1,  # default value
            gamma=0.99
        )
    
    # add the global step counter
    scheduler.global_step = 0
    return scheduler

#############################################################################
# Model Training/Validation/Testing
#############################################################################

def train_model(model,train_loader,optimizer,scheduler,loss_fn,reg_fn=None,device="cpu",model_name="DispFormer",args=None):
    model.train()
    train_loss = 0.0
    step_loss = []
    
    if not hasattr(scheduler, 'global_step'):
        scheduler.global_step = 0  # initialize the global step
    
    # Create a nested progress bar for steps within this epoch
    step_pbar = tqdm(total=len(train_loader), desc=f"Training Steps", leave=False, position=1)

    for batch_idx, (input_data, input_mask, target_data, used_layer) in enumerate(train_loader):      
        # input data
        if model_name.lower() in ["unet","dispformer","dispformer_local_global","dispformer_local_global_v1","dispformer_local_global_v2"]:
            # period, phase velocity, group velocity
            input_data  = input_data.to(device)
            input_mask  = input_mask.to(device)
            target_data = target_data.to(device)
        elif model_name.lower() in ["sfnet"]:
            # phase velocity, group velocity
            input_data  = input_data[:,1:,:].to(device)
            target_data = target_data.to(device)
        elif model_name.lower() in ["fcnn"]:
            # period, phase velocity
            input_data  = input_data[:,0:2,:].to(device)
            target_data = target_data.to(device)
        
        # model forward
        optimizer.zero_grad()
        if model_name.lower() in ["dispformer","dispformer_local_global","dispformer_local_global_v1","dispformer_local_global_v2"]:
            output = model(input_data, input_mask)
        elif model_name.lower() in ["sfnet","fcnn","unet"]:
            if args is not None and args.pretrained_model_path is not None:
                # padding the input data to the seq_length with -1 [B,v,L1] -> [B,v,Seq_len]
                seq_length = args.seq_length
                input_data = F.pad(input_data, (0, seq_length-input_data.shape[-1]), "constant", -1)
            output = model(input_data)
        
        # misfit calculation
        loss = 0.0
        for i in range(used_layer.shape[0]):
            loss += loss_fn(output[i][used_layer[i,0]:used_layer[i,1]], target_data[i,1,used_layer[i,0]:used_layer[i,1]])
            
            # w1,w2 = 1,1/10
            # loss_used_layer = loss_fn(output[i][used_layer[i,0]:used_layer[i,1]], target_data[i,1,used_layer[i,0]:used_layer[i,1]])
            # loss_other_layer_left = loss_fn(output[i][0:used_layer[i,0]], target_data[i,1,0:used_layer[i,0]]) if used_layer[i,0] > 0 else torch.tensor(0.0)
            # loss_other_layer_right = loss_fn(output[i][used_layer[i,1]:], target_data[i,1,used_layer[i,1]:]) if used_layer[i,1] < target_data.shape[2] else torch.tensor(0.0)
            # loss += (w1*loss_used_layer + w2*(loss_other_layer_left + loss_other_layer_right))
            
            if reg_fn is not None:
                reg_loss = reg_fn(output[i][used_layer[i,0]:used_layer[i,1]])
                loss += reg_loss
            
        loss = loss / used_layer.shape[0]
        
        # backward and update
        loss.backward()
        optimizer.step()
        
        # update the scheduler
        if args is not None and args.scheduler_method.lower() in ["steplr+warmup","cosine+warmup"]:
            scheduler.step()
            scheduler.global_step += 1
        
        train_loss += loss.item()
        step_loss.append(loss.item())
        
        if batch_idx % 1000 == 0:
            step_pbar.set_description(f"Step: {batch_idx+1}, Loss: {loss.item():.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")
        
    # update the learning rate when the scheduler is [StepLR,CosineAnnealingLR]
    if args is None:
        scheduler.step()
    elif args is not None and args.scheduler_method.lower() in ["steplr","cosine"]:
        scheduler.step()
    elif args is not None and args.scheduler_method.lower() in ["steplr+warmup","cosine+warmup"]:
        pass
    
    train_loss = train_loss / len(train_loader)
    return train_loss,step_loss


def valid_model(model,val_loader,loss_fn,device="cpu",model_name="DispFormer",args=None):
    # consider the operation in Dispformer/Dispformer-LG, we use the train mode to calculate the validation loss
    # model.eval()
    model.train()
    
    val_loss = 0.0
    step_loss = []
    with torch.no_grad():
        for batch_idx, (input_data, input_mask, target_data, used_layer) in enumerate(val_loader):
            # input data
            if model_name.lower() in ["unet","dispformer","dispformer_local_global","dispformer_local_global_v1","dispformer_local_global_v2"]:
                input_data = input_data.to(device)
                input_mask = input_mask.to(device)
                target_data = target_data.to(device)
            elif model_name.lower() in ["sfnet"]:
                input_data = input_data[:,1:,:].to(device)
                target_data = target_data.to(device)
            elif model_name.lower() in ["fcnn"]:
                input_data = input_data[:,0:2,:].to(device)
                target_data = target_data.to(device)
            
            # model forward
            if model_name.lower() in ["dispformer","dispformer_local_global","dispformer_local_global_v1","dispformer_local_global_v2"]:
                output = model(input_data, input_mask)
            elif model_name.lower() in ["sfnet","fcnn","unet"]:
                if args is not None and args.pretrained_model_path is not None:
                    # padding the input data to the seq_length with -1 [B,v,L1] -> [B,v,Seq_len]
                    seq_length = args.seq_length
                    input_data = F.pad(input_data, (0, seq_length-input_data.shape[-1]), "constant", -1)
                output = model(input_data)
            
            # misfit calculation
            loss = 0.0
            for i in range(used_layer.shape[0]):
                loss += loss_fn(output[i][used_layer[i,0]:used_layer[i,1]], target_data[i,1,used_layer[i,0]:used_layer[i,1]])

            loss = loss / used_layer.shape[0]
            val_loss += loss.item()
            step_loss.append(loss.item())   
            
    val_loss = val_loss / len(val_loader)
    return val_loss,step_loss

def test_model(model,test_loader,loss_fn,device="cpu",model_name="DispFormer"):
    # model.eval()
    model.train()
    test_loss = 0.0
    test_targets,test_outputs,test_used_layer = [],[],[]
    with torch.no_grad():
        for batch_idx, (input_data, input_mask, target_data, used_layer) in enumerate(test_loader):
            # input data
            if model_name.lower() in ["unet","dispformer","dispformer_local_global","dispformer_local_global_v1","dispformer_local_global_v2"]:
                # period, phase velocity, group velocity
                input_data = input_data.to(device)  
                input_mask = input_mask.to(device)
                target_data = target_data.to(device)
            elif model_name.lower() in ["sfnet"]:
                # phase velocity, group velocity
                input_data = input_data[:,1:,:].to(device)
                target_data = target_data.to(device)
            elif model_name.lower() in ["fcnn"]:
                # period, phase velocity
                input_data = input_data[:,0:2,:].to(device)
                target_data = target_data.to(device)
            
            # model forward
            if model_name.lower() in ["dispformer","dispformer_local_global","dispformer_local_global_v1","dispformer_local_global_v2"]:
                output = model(input_data, input_mask)
            elif model_name.lower() in ["sfnet","fcnn","unet"]:
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
    
    # initialize the model
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
    elif model_name.lower() == "dispformer_local_global_v1":
        model = DispFormer_local_global_v1(model_dim=model_dim, 
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
    
    # load trained parameters for model evaluation
    if load_trained_model:
        save_path,save_path_name = get_save_path_name(args)
        saved_model_path = os.path.join(save_path,"best_model.pth")
    else:
        saved_model_path = None
    
    if saved_model_path is not None:
        model.load_state_dict(torch.load(saved_model_path,map_location=device))
    return model