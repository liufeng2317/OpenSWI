import argparse
import os

def str2bool(str):
	return True if str.lower() == 'true' else False

# get the arguments from the settings and the command line
def parse_args(settings,jupyter=False):
    parser = argparse.ArgumentParser()
    # data
    if "input_data_path" in settings["data"]:
        parser.add_argument("--input_data_path", type=str, default=settings["data"]["input_data_path"])
    if "target_data_path" in settings["data"]:
        parser.add_argument("--target_data_path", type=str, default=settings["data"]["target_data_path"])
    if "train" in settings["data"]:
        parser.add_argument("--train", type=str2bool, default=settings["data"]["train"])
    # interpolation velocity model
    if "interp_layer" in settings["data"]:
        parser.add_argument("--interp_layer", type=str2bool, default=settings["data"]["interp_layer"])
    if "interp_layer_thickness" in settings["data"]:
        parser.add_argument("--interp_layer_thickness", type=float, default=settings["data"]["interp_layer_thickness"])
    if "interp_layer_number" in settings["data"]:
        parser.add_argument("--interp_layer_number", type=int, default=settings["data"]["interp_layer_number"])
    if "interp_kind" in settings["data"]:
        parser.add_argument("--interp_kind", type=str, default=settings["data"]["interp_kind"])
    if "layer_used_range" in settings["data"]:
        parser.add_argument("--layer_used_range", type=str, default=settings["data"]["layer_used_range"])
    if "normalize_input_data" in settings["data"]:
        parser.add_argument("--normalize_input_data", type=str2bool, default=settings["data"]["normalize_input_data"])    
    # augmentation for training data
    if "aug_train_data" in settings["data"]:
        parser.add_argument("--aug_train_data", type=str2bool, default=settings["data"]["aug_train_data"])
    if "aug_noise_level" in settings["data"]:
        parser.add_argument("--aug_noise_level", type=float, default=settings["data"]["aug_noise_level"])
    if "aug_random_mask_ratio" in settings["data"]:
        parser.add_argument("--aug_random_mask_ratio", type=float, default=settings["data"]["aug_random_mask_ratio"])
    if "aug_max_masking_length" in settings["data"]:
        parser.add_argument("--aug_max_masking_length", type=int, default=settings["data"]["aug_max_masking_length"])
    if "aug_remove_group_ratio" in settings["data"]:
        parser.add_argument("--aug_remove_group_ratio", type=float, default=settings["data"]["aug_remove_group_ratio"])
    if "aug_remove_phase_ratio" in settings["data"]:
        parser.add_argument("--aug_remove_phase_ratio", type=float, default=settings["data"]["aug_remove_phase_ratio"])
    # augmentation for pretraining data
    if "aug_varylength" in settings["data"]:
        parser.add_argument("--aug_varylength", type=str2bool, default=settings["data"]["aug_varylength"])
    if "aug_varylength_start_range" in settings["data"]:
        parser.add_argument("--aug_varylength_start_range", type=str, default=settings["data"]["aug_varylength_start_range"])
    if "aug_varylength_min_length" in settings["data"]:
        parser.add_argument("--aug_varylength_min_length", type=int, default=settings["data"]["aug_varylength_min_length"])
    if "aug_varylength_c1" in settings["data"]:
        parser.add_argument("--aug_varylength_c1", type=float, default=settings["data"]["aug_varylength_c1"])
    if "aug_varylength_c2" in settings["data"]:
        parser.add_argument("--aug_varylength_c2", type=float, default=settings["data"]["aug_varylength_c2"])
    if "aug_varylength_c3" in settings["data"]:
        parser.add_argument("--aug_varylength_c3", type=float, default=settings["data"]["aug_varylength_c3"])
    
    if "random_seed" in settings["data"]:
        parser.add_argument("--random_seed", type=int, default=settings["data"]["random_seed"])
    if "batch_size" in settings["data"]:
        parser.add_argument("--batch_size", type=int, default=settings["data"]["batch_size"])
    if "num_workers" in settings["data"]:
        parser.add_argument("--num_workers", type=int, default=settings["data"]["num_workers"])

    # model
    # pretrained model
    if "pretrained_model_path" in settings["model"]:
        parser.add_argument("--pretrained_model_path", type=str, default=settings["model"]["pretrained_model_path"] if settings["model"]["pretrained_model_path"] != "" else None)
    else:
        parser.add_argument("--pretrained_model_path", type=str, default=None)
    if "model_name" in settings["model"]:
        parser.add_argument("--model_name", type=str, default=settings["model"]["model_name"])
    if "output_dim" in settings["model"]:
        parser.add_argument("--output_dim", type=int, default=settings["model"]["output_dim"])
    if "scale_factor" in settings["model"]:
        parser.add_argument("--scale_factor", type=float, default=settings["model"]["scale_factor"])    
    # DispFormer/DispFormer_local_global
    if "model_dim" in settings["model"]:
        parser.add_argument("--model_dim", type=int, default=settings["model"]["model_dim"])
    if "num_heads" in settings["model"]:
        parser.add_argument("--num_heads", type=int, default=settings["model"]["num_heads"])
    if "num_layers" in settings["model"]:
        parser.add_argument("--num_layers", type=int, default=settings["model"]["num_layers"])    
    # SfNet
    if "seq_length" in settings["model"]:
        parser.add_argument("--seq_length", type=int, default=settings["model"]["seq_length"])
    if "init_features" in settings["model"]:
        parser.add_argument("--init_features", type=int, default=settings["model"]["init_features"])

    # training
    if "checkpoints_path" in settings["training"]:
        parser.add_argument("--checkpoints_path", type=str, default=settings["training"]["checkpoints_path"])
    if "results_path" in settings["training"]:
        parser.add_argument("--results_path", type=str, default=settings["training"]["results_path"])
    # data and physical loss
    if "loss_type" in settings["training"]:
        parser.add_argument("--loss_type", type=str, default=settings["training"]["loss_type"])
    if "phy_loss" in settings["training"]:
        parser.add_argument("--phy_loss", type=str2bool, default=settings["training"]["phy_loss"])
    if "phy_loss_weight" in settings["training"]:
        parser.add_argument("--phy_loss_weight", type=float, default=settings["training"]["phy_loss_weight"])
    # regularization
    if "reg_type" in settings["training"]:
        parser.add_argument("--reg_type", type=str, default=settings["training"]["reg_type"] if settings["training"]["reg_type"] != "" else None)
    else:
        parser.add_argument("--reg_type", type=str, default=None)
    if "reg_weight" in settings["training"]:
        parser.add_argument("--reg_weight", type=float, default=settings["training"]["reg_weight"])
    # training parameters
    if "lr" in settings["training"]:
        parser.add_argument("--lr", type=float, default=settings["training"]["lr"])
    if "num_epochs" in settings["training"]:
        parser.add_argument("--num_epochs", type=int, default=settings["training"]["num_epochs"])
    if "scheduler_method" in settings["training"]:
        parser.add_argument("--scheduler_method", type=str, default=settings["training"]["scheduler_method"] if settings["training"]["scheduler_method"] != "" else "StepLR")
    else:
        parser.add_argument("--scheduler_method", type=str, default="StepLR")
    if "warmup_ratio" in settings["training"]:
        parser.add_argument("--warmup_ratio", type=float, default=settings["training"]["warmup_ratio"])
    if "step_size" in settings["training"]:
        parser.add_argument("--step_size", type=int, default=settings["training"]["step_size"])
    if "gamma" in settings["training"]:
        parser.add_argument("--gamma", type=float, default=settings["training"]["gamma"])
    if "early_stopping_flag" in settings["training"]:
        parser.add_argument("--early_stopping_flag", type=str2bool, default=settings["training"]["early_stopping_flag"])
    else:
        parser.add_argument("--early_stopping_flag", type=str2bool, default=False)
    if "early_stopping_patience" in settings["training"]:
        parser.add_argument("--early_stopping_patience", type=int  , default=settings["training"]["early_stopping_patience"])
    if "save_epochs" in settings["training"]:
        parser.add_argument("--save_epochs", type=int, default=settings["training"]["save_epochs"])
    if "device" in settings["training"]:
        parser.add_argument("--device", type=str, default=settings["training"]["device"])
    
    if jupyter:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args

# update the settings with the arguments
def update_settings(settings,args):
    # data
    if "input_data_path" in settings["data"]:
        settings["data"]["input_data_path"]  = args.input_data_path
    if "target_data_path" in settings["data"]:
        settings["data"]["target_data_path"] = args.target_data_path
    if "train" in settings["data"]:
        settings["data"]["train"] = args.train
    if "interp_layer" in settings["data"]:
        settings["data"]["interp_layer"] = args.interp_layer
    if "interp_layer_thickness" in settings["data"]:
        settings["data"]["interp_layer_thickness"] = args.interp_layer_thickness
    if "interp_layer_number" in settings["data"]:
        settings["data"]["interp_layer_number"] = args.interp_layer_number
    if "interp_kind" in settings["data"]:
        settings["data"]["interp_kind"] = args.interp_kind
    if "layer_used_range" in settings["data"]:
        settings["data"]["layer_used_range"] = args.layer_used_range
    if "normalize_input_data" in settings["data"]:
        settings["data"]["normalize_input_data"] = args.normalize_input_data
    # augmentation for training data
    if "aug_train_data" in settings["data"]:
        settings["data"]["aug_train_data"] = args.aug_train_data
    if "aug_noise_level" in settings["data"]:
        settings["data"]["aug_noise_level"] = args.aug_noise_level
    if "aug_random_mask_ratio" in settings["data"]:
        settings["data"]["aug_random_mask_ratio"] = args.aug_random_mask_ratio
    if "aug_max_masking_length" in settings["data"]:
        settings["data"]["aug_max_masking_length"] = args.aug_max_masking_length
    if "aug_remove_group_ratio" in settings["data"]:
        settings["data"]["aug_remove_group_ratio"] = args.aug_remove_group_ratio
    if "aug_remove_phase_ratio" in settings["data"]:
        settings["data"]["aug_remove_phase_ratio"] = args.aug_remove_phase_ratio
    # augmentation for pretraining data
    if "aug_varylength" in settings["data"]:
        settings["data"]["aug_varylength"] = args.aug_varylength
    if "aug_varylength_start_range" in settings["data"]:
        settings["data"]["aug_varylength_start_range"] = args.aug_varylength_start_range
    if "aug_varylength_min_length" in settings["data"]:
        settings["data"]["aug_varylength_min_length"] = args.aug_varylength_min_length
    if "aug_varylength_c1" in settings["data"]:
        settings["data"]["aug_varylength_c1"] = args.aug_varylength_c1
    if "aug_varylength_c2" in settings["data"]:
        settings["data"]["aug_varylength_c2"] = args.aug_varylength_c2
    if "aug_varylength_c3" in settings["data"]:
        settings["data"]["aug_varylength_c3"] = args.aug_varylength_c3

    if "random_seed" in settings["data"]:
        settings["data"]["random_seed"]      = args.random_seed
    if "batch_size" in settings["data"]:
        settings["data"]["batch_size"] = args.batch_size
    if "num_workers" in settings["data"]:
        settings["data"]["num_workers"] = args.num_workers
    
    # model
    if "pretrained_model_path" in settings["model"]:
        settings["model"]["pretrained_model_path"] = args.pretrained_model_path
    if "model_name" in settings["model"]:
        settings["model"]["model_name"] = args.model_name
    if "output_dim" in settings["model"]:
        settings["model"]["output_dim"] = args.output_dim
    if "scale_factor" in settings["model"]:
        settings["model"]["scale_factor"] = args.scale_factor
    # model (DispFormer)
    if "model_dim" in settings["model"]:
        settings["model"]["model_dim"] = args.model_dim
    if "num_heads" in settings["model"]:
        settings["model"]["num_heads"] = args.num_heads
    if "num_layers" in settings["model"]:
        settings["model"]["num_layers"] = args.num_layers
    # model (SfNet)
    if "seq_length" in settings["model"]:
        settings["model"]["seq_length"] = args.seq_length
    if "init_features" in settings["model"]:
        settings["model"]["init_features"] = args.init_features
        
    # training
    if "checkpoints_path" in settings["training"]:
        settings["training"]["checkpoints_path"] = args.checkpoints_path
    if "results_path" in settings["training"]:
        settings["training"]["results_path"] = args.results_path
    if "loss_type" in settings["training"]:
        settings["training"]["loss_type"] = args.loss_type
    if "phy_loss" in settings["training"]:
        settings["training"]["phy_loss"] = args.phy_loss
    if "phy_loss_weight" in settings["training"]:
        settings["training"]["phy_loss_weight"] = args.phy_loss_weight
    if "reg_type" in settings["training"]:
        settings["training"]["reg_type"] = args.reg_type
    if "reg_weight" in settings["training"]:
        settings["training"]["reg_weight"] = args.reg_weight
    if "lr" in settings["training"]:
        settings["training"]["lr"] = args.lr
    if "num_epochs" in settings["training"]:
        settings["training"]["num_epochs"] = args.num_epochs
    if "save_epochs" in settings["training"]:
        settings["training"]["save_epochs"] = args.save_epochs
    if "scheduler_method" in settings["training"]:
        settings["training"]["scheduler_method"] = args.scheduler_method
    if "warmup_ratio" in settings["training"]:
        settings["training"]["warmup_ratio"] = args.warmup_ratio
    if "step_size" in settings["training"]:
        settings["training"]["step_size"] = args.step_size
    if "gamma" in settings["training"]:
        settings["training"]["gamma"] = args.gamma
    if "device" in settings["training"]:
        settings["training"]["device"] = args.device
    if "early_stopping_flag" in settings["training"]:
        settings["training"]["early_stopping_flag"] = args.early_stopping_flag
    if "early_stopping_patience" in settings["training"]:
        settings["training"]["early_stopping_patience"] = args.early_stopping_patience
    
    # save path
    
    return settings

def get_save_path_name(args):
    
    # for the finetune model
    if args.pretrained_model_path is not None:
        # Extract model name from path, handling both Windows and Unix-style paths
        pretrained_model_path = os.path.basename(os.path.dirname(args.pretrained_model_path))
        pretrained_model_path = "pretrained_"+pretrained_model_path
    
    # for the normal-training model
    if args.model_name.lower() in ["dispformer","dispformer_local_global","dispformer_local_global_v1","dispformer_local_global_v2"]:
        save_path_name = f"{args.model_name}_md={args.model_dim}_nh={args.num_heads}_nl={args.num_layers}_od={args.output_dim}_lt={args.loss_type}_lr={args.lr}_ne={args.num_epochs}_bs={args.batch_size}_ni={args.normalize_input_data}"
    elif args.model_name.lower() in ["sfnet","unet"]:
        save_path_name = f"{args.model_name}_sl={args.seq_length}_if={args.init_features}_od={args.output_dim}_sf={args.scale_factor}_lt={args.loss_type}_lr={args.lr}_ne={args.num_epochs}_bs={args.batch_size}_ni={args.normalize_input_data}"
    elif args.model_name.lower() in ["fcnn"]:
        save_path_name = f"{args.model_name}_sl={args.seq_length}_od={args.output_dim}_sf={args.scale_factor}_lt={args.loss_type}_lr={args.lr}_ne={args.num_epochs}_bs={args.batch_size}_ni={args.normalize_input_data}"

    # the save path
    if args.pretrained_model_path is not None:
        save_path = os.path.join(args.checkpoints_path, args.model_name, pretrained_model_path, save_path_name)
    else:
        save_path = os.path.join(args.checkpoints_path, args.model_name, save_path_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path,save_path_name