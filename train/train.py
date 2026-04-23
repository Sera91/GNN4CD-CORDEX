import torch
import numpy as np
import pickle
import time
import argparse
import os
import importlib
import json
import random
from accelerate import Accelerator

from utils.losses.qmse import derive_qmse_bins
from utils.helpers import write_log, inspect_model, set_seed_everything
from utils.helpers import find_not_all_nan_times, derive_train_val_idxs, derive_train_val_idxs_years_list
from utils.helpers import prepare_target_for_train
from utils.training import Trainer
from data.datasets import Graph_Dataset, custom_collate_fn_graph
from models import MODEL_REGISTRY, build_model, update_parser_with_model_args
from utils.predictand_transforms import transform_predictand
from utils.predictor_transforms import transform_predictors
from utils.losses import LOSS_REGISTRY, build_loss, update_parser_with_loss_args

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--low_input_file', type=str, default=None)
parser.add_argument('--orog_file', type=str, default=None)
parser.add_argument('--mask_sealand_file', type=str, default=None)
parser.add_argument('--target_file', type=str, default=None)
parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--coords_ij_file', type=str, default=None)

parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')
parser.add_argument('--wandb_project_name', type=str)

parser.add_argument('--metadata_file', type=str, help='metadata file')

#-- training hyperparameters
parser.add_argument('--epochs', type=int, default=15, help='number of total training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (global)')
parser.add_argument('--step_size', type=int, default=10, help='scheduler step size (global)')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (wd)')
parser.add_argument('--load_checkpoint',  action='store_true')
parser.add_argument('--no-load_checkpoint', dest='load_checkpoint', action='store_false')
parser.add_argument('--lr_scheduler', type=str, default="StepLR")

parser.add_argument('--checkpoint_ctd', type=str, help='checkpoint to load to continue')
parser.add_argument('--ctd_training',  action='store_true')
parser.add_argument('--no-ctd_training', dest='ctd_training', action='store_false')
parser.add_argument('--make_val_plots', action='store_true')
parser.add_argument('--no-make_val_plots', dest='make_val_plots', action='store_false')
parser.add_argument('--val_plot_frequency', type=int)

parser.add_argument('--loss_fn', type=str)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--n_gpu', type=int, default=4)

parser.add_argument('--model_name', type=str)
parser.add_argument('--history_length', type=str)

parser.add_argument('--predictand_transform_mode', type=str)
parser.add_argument('--predictor_low_tranform_mode', type=str)
parser.add_argument('--predictor_high_tranform_mode', type=str)

parser.add_argument('--dataset_name', type=str, default='graph_dataset')
parser.add_argument('--collate_name', type=str)
parser.add_argument('--target_type', type=str)

#-- start and end training dates
parser.add_argument('--train_year_start', type=str, default="")
parser.add_argument('--train_month_start', type=str, default="")
parser.add_argument('--train_day_start', type=str, default="")
parser.add_argument('--train_year_end', type=str, default="")
parser.add_argument('--train_month_end', type=str, default="")
parser.add_argument('--train_day_end', type=str, default="")
parser.add_argument('--validation_year', type=str, default="")
# for random validation years
parser.add_argument('--first_year', type=str, default="")
parser.add_argument('--last_year', type=str, default="")
parser.add_argument('--n_val_years', type=str, default="")
# for lists of training and validation years
parser.add_argument('--train_years', type=str, default="")
parser.add_argument('--val_years', type=str, default="")

parser.add_argument('--WANDB_API_KEY', type=str)
parser.add_argument('--WANDB_USERNAME', type=str)

if __name__ == '__main__':

    args = parser.parse_args()

    # Set all seeds
    set_seed_everything(seed=args.seed)
    #torch.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


#-----------------------------------------------------
#--------------- WANDB and ACCELERATE ----------------
#-----------------------------------------------------

    if args.use_accelerate is True:
        accelerator = Accelerator(log_with="wandb", step_scheduler_with_optimizer=False)
    else:
        accelerator = None
    
    os.environ['WANDB_API_KEY'] = args.WANDB_API_KEY
    os.environ['WANDB_USERNAME'] = args.WANDB_USERNAME
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_CONFIG_DIR']='./wandb/'
    os.environ['WANDB_SERVICE_WAIT'] = '300'

    accelerator.init_trackers(
            project_name=args.wandb_project_name
        )

    write_log(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.", args, accelerator, 'w')

#--------------------------------------------------------
#--------------------- LOAD FILES -----------------------
#--------------------------------------------------------

    #-- 1. Graph
    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    #-- 2. Low res input
    x_low = np.load(args.input_path+args.low_input_file)

    #-- 3. Target
    target = np.load(args.input_path+args.target_file)

    #-- 4. Orography
    orog = np.load(args.input_path+args.orog_file)

    #-- 5. Mask sea-land
    if args.mask_sealand_file != "":
        mask_sealand = np.load(args.input_path+args.mask_sealand_file)
        use_mask_sealand = True
    else:
        use_mask_sealand = False
    
    #-- 6. Coords ij
    if args.coords_ij_file != "":
        coords_ij = np.load(args.input_path+args.coords_ij_file)
        use_coords_ij = True
    else:
        use_coords_ij = False

    #-- 7. Time index
    time_index = np.load(args.input_path+"time_index.npy")

    #-- 8. Low input metadata
    with open(args.input_path + args.metadata_file, "r") as f:
        metadata = json.load(f)

#--------------------------------------------------------
#-------------------  TRAIN/VAL IDXS --------------------
#--------------------------------------------------------

    #-- Step 1 - Find valid time indices
    idxs_not_all_nan = find_not_all_nan_times(target)

    write_log(f"\nAfter removing all nan time indexes, {len(idxs_not_all_nan)}" +
        f" time indexes are considered ({(len(idxs_not_all_nan) / target.shape[1] * 100):.1f} % of initial ones).",
        args, accelerator, 'a')

    #-- Step 2 - Compute train/val indices
    # Case 1: the user provides lists of train years and val years
    write_log(f"\nargs.train_years: {args.train_years.split(' ')}\nargs.val_years: {args.val_years.split(' ')}", args, accelerator, 'a')
    if args.train_years != "" and args.val_years != "":
        train_years = [int(y) for y in args.train_years.split(' ')]
        val_years = [int(y) for y in args.val_years.split(' ')]
        train_idxs, train_idxs_valid_subset, val_idxs, val_idxs_valid_subset = derive_train_val_idxs_years_list(
            train_years,
            val_years,
            history_length=args.history_length,
            time_index=time_index,
            args=args,
            accelerator=accelerator
        )
    elif args.first_year != "" and args.last_year != "":
        # Build the full range
        all_years = list(range(int(args.first_year), int(args.last_year) + 1))
        # Randomly sample validation years
        val_years = random.sample(all_years, int(args.n_val_years))
        # Training years are the complement
        train_years = [y for y in all_years if y not in val_years]
        train_idxs, train_idxs_valid_subset, val_idxs, val_idxs_valid_subset = derive_train_val_idxs_years_list(
            train_years,
            val_years,
            history_length=args.history_length,
            time_index=time_index,
            args=args,
            accelerator=accelerator
        )
    else:
        train_idxs, train_idxs_valid_subset, val_idxs, val_idxs_valid_subset = derive_train_val_idxs(
            int(args.train_year_start), int(args.train_month_start), int(args.train_day_start),
            int(args.train_year_end), int(args.train_month_end), int(args.train_day_end),
            history_length=args.history_length,
            time_index=time_index,
            idxs_not_all_nan=idxs_not_all_nan,
            validation_year=int(args.validation_year),
            args=args,
            accelerator=accelerator
        )

    np.save(args.output_path + "train_idxs.npy", train_idxs)
    np.save(args.output_path + "train_idxs_valid_subset.npy", train_idxs_valid_subset)
    np.save(args.output_path + "val_idxs.npy", val_idxs)
    np.save(args.output_path + "val_idxs_valid_subset.npy", val_idxs_valid_subset)

#--------------------------------------------------------
#---------  TRANSFORM PREDICTORS AND PREDICTAND ---------
#--------------------------------------------------------

    # 1. Transform predictors
    x_low_std, x_high_std = transform_predictors(
        x_low,
        x_high=orog,
        train_idxs=train_idxs,
        mode_low=args.predictor_low_tranform_mode,      # e.g. "zscore_lowres_var"
        mode_high=args.predictor_high_transform_mode,    # e.g. "zscore_highres_grouped"
        stats=None,
        stats_save_path=args.output_path + "predictors_stats.npz"
    )

    n_vars = x_low_std.shape[2]
    n_levels = x_low_std.shape[3]

    # 1.2 Flatten x_low_std
    x_low_std = torch.flatten(x_low_std, start_dim=2, end_dim=-1)   # num_nodes, time, vars*levels

    # 1.3 Add high_res predictors which are not transformed
    if use_mask_sealand:
        x_high_std = np.concatenate((x_high_std, mask_sealand), axis=-1)
        write_log(f"\nAdding mask sea-land node features", args, accelerator, 'a')

    if use_coords_ij:
        x_high_std = np.concatenate((x_high_std, coords_ij), axis=-1)
        write_log(f"\nAdding ij node features", args, accelerator, 'a')

    n_static_high = x_high_std.shape[1]

    write_log(f"\nn_vars: {n_vars}, n_levels: {n_levels}, n_static_high: {n_static_high}", args, accelerator, 'a')

    # 2. Transform predictand
    if args.loss_name == "BernoulliGammaNLLLoss":
        target_trans = target
    else:
        target_trans = transform_predictand(
            target,
            mode=args.predictand_transform_mode,      # e.g. "log1p", "z_score", "minmax"
            stats_save_path=args.output_path + "predictand_stats.npz"
        )
    
    #-----------------------------------------------------
    #-------------- BUILD LOSS and MODEL -----------------
    #-----------------------------------------------------

    # Update args with loss- and model-specific arguments
    parser = update_parser_with_loss_args(args.loss_name)
    parser = update_parser_with_model_args(args.model_name)
    args = parser.parse_args()

    loss, output_dim = build_loss(args)
    
    model = build_model(
        x_low_var_dim=n_vars,
        x_low_lev_dim=n_levels,
        x_high_dim=n_static_high,
        output_dim=output_dim,
        args=args
    )

    # Eventually compute QMSE bins
    if args.loss_name == "MSE_QMSE_PSD_Loss":
        if args.binscale == "log":
            binmin = np.log1p(args.binmin)
            binmax = np.log1p(args.binmax)
            binwidth = np.log1p(args.binwidth)
        else:
            binmin = args.binmin
            binmax = args.binmax
            binwidth = args.binwidth

        target_bins = derive_qmse_bins(
            target_trans,
            train_idxs[train_idxs_valid_subset],
            args,
            accelerator,
            binmin=binmin,
            binmax=binmax,
            binwidth=binwidth
        )
    
    #-----------------------------------------------------
    #---------------- SPLIT IN TRAIN/VAL -----------------
    #-----------------------------------------------------

    x_low_std_train = x_low_std[:, train_idxs, :]
    x_low_std_val = x_low_std[:, val_idxs, :]

    target_trans_train = target_trans[:, train_idxs]
    target_trans_val = target_trans[:, val_idxs]

    if args.loss_name == "MSE_QMSE_PSD_Loss":
        target_bins_train = target_bins[:, train_idxs]
        target_bins_val = target_bins[:, val_idxs]

    #-----------------------------------------------------
    #-------------- FROM NUMPY TO PYTORCH ----------------
    #-----------------------------------------------------

    # Predictors
    x_low_std_train = torch.from_numpy(x_low_std_train).float()
    x_low_std_val = torch.from_numpy(x_low_std_val).float()
    x_high_std = torch.from_numpy(x_high_std).float()

    # Predictand and eventually QMSE bins
    target_trans_train = torch.from_numpy(target_trans_train).float()
    target_trans_val = torch.from_numpy(target_trans_val).float()

    if args.loss_name == "MSE_QMSE_PSD_Loss":
        target_bins_train = torch.from_numpy(target_bins_train).int()
        target_bins_val = torch.from_numpy(target_bins_val).int()

    #-----------------------------------------------------
    #-------------- DATASET AND DATALOADER ---------------
    #-----------------------------------------------------
  
    # Create two different datasets for efficiency
    graph_dataset_train_tmp = Graph_Dataset(
        low_high_graph,
        x_low_std_train,
        x_high_std,
        target_trans_train,
        args.history_length,
    )

    if args.validation_year is not None or args.val_years is not None:
        graph_dataset_val_tmp = Graph_Dataset(
            low_high_graph,
            x_low_std_val,
            x_high_std,
            target_trans_val,
            args.history_length,
        )

    if args.loss_name == "MSE_QMSE_PSD_Loss":
        graph_dataset_train_tmp.set_additional_features(w=target_bins_train)
        graph_dataset_val_tmp.set_additional_features(w=target_bins_val)

    graph_dataset_train = torch.utils.data.Subset(graph_dataset_train_tmp, train_idxs_valid_subset) # it's just a view of the original dataset
    graph_dataset_val = torch.utils.data.Subset(graph_dataset_val_tmp, val_idxs_valid_subset)
        
    # len(graph_dataset_train) will be the number of training exaples (inputs are bigger, considering the history length)
    write_log(f'\nTrainset size = {len(graph_dataset_train)}, validationset size = {len(graph_dataset_val)}.', args, accelerator, 'a')

    # Define the dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        graph_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn_graph,
        num_workers=0
    )

    dataloader_val = torch.utils.data.DataLoader(
        graph_dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn_graph,
        num_workers=0
    )

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        write_log(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %", args, accelerator, 'a')

    #-----------------------------------------------------
    #------------ OPTIMIZER AND LR SCHEDULER -------------
    #-----------------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    elif args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6, last_epoch=-1)
    else:
        lr_scheduler = None

#-----------------------------------------------------
#---------------- ACCELERATE PREPARE -----------------
#-----------------------------------------------------

    epoch_start=0
    
    if accelerator is not None:
        model, optimizer, dataloader_train, lr_scheduler, loss = accelerator.prepare(
            model, optimizer, dataloader_train, lr_scheduler, loss)
        dataloader_val = accelerator.prepare(dataloader_val)
        write_log("\nUsing accelerator to prepare model, optimizer, dataloader and loss...", args, accelerator, 'a')
    else:
        write_log("\nNot using accelerator to prepare model, optimizer, dataloader and loss...", args, accelerator, 'a')
        model = model.cuda()

    if args.ctd_training:
        write_log("\nContinuing the training.")
        accelerator.load_state(args.checkpoint_ctd)
        epoch_start = torch.load(args.checkpoint_ctd+"epoch")["epoch"] + 1
    
    inspect_model(model, args, accelerator)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(f"\nTotal number of trainable parameters: {total_params}.", args, accelerator, 'a')

#-----------------------------------------------------
#----------------------- TRAIN -----------------------
#-----------------------------------------------------

    write_log(f"\nUsing lr={optimizer.param_groups[0]['lr']:.8f}, " +
                f"weight decay = {args.weight_decay} and epochs={args.epochs}." + 
                f"\nloss: {loss}", args, accelerator, 'a') 
    
    effective_batch_size = args.batch_size if accelerator is None else args.batch_size*torch.cuda.device_count()
    write_log(f"\nModel = {args.model_name}, batch size = {effective_batch_size}", args, accelerator, 'a')

    val_size = len(graph_dataset_val)

    start = time.time()    

    Trainer.train(
        model,
        dataloader_train,
        dataloader_val,
        optimizer,
        loss,
        lr_scheduler,
        val_size,
        time_index[val_idxs][val_idxs_valid_subset],
        accelerator,
        args,
        epoch_start=epoch_start)

    end = time.time()

    write_log(f"\nCompleted in {end - start} seconds.\nDONE!", args, accelerator, 'a')
    

