import torch
import numpy as np
import pickle
import time
import wandb
from utils.metrics.metrics import AverageMeter
from utils.helpers.tools import write_log, invert_normalization
from utils.plotting.plots import plot_maps, plot_pdf, get_cmap_dict
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib
import json
from utils.helpers.tools import convert_dict
from torch_geometric.data import HeteroData
from utils.plotting import create_validation_plots

#-----------------------------------------------------
#---------------------- TRAIN ------------------------
#-----------------------------------------------------

class Trainer(object):

    def __init__(self):
        super(Trainer, self).__init__()

    def train_reg(
            self,
            model,
            dataloader_train,
            dataloader_val,
            optimizer,
            loss_fn,
            lr_scheduler,
            val_size,
            times,
            accelerator,
            args,
            epoch_start=0,
            log_val_plots=True):
        
        write_log(f"\nStart training the regressor.", args, accelerator, 'a')

        step = 0
        
        for epoch in range(epoch_start, epoch_start+args.epochs):

            model.train()
            write_log(f"\nEpoch {epoch} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')
            
            # Define objects to track meters
            loss_meter = AverageMeter()
            val_loss_meter = AverageMeter()

            start = time.time()
            
            # TRAIN
            for i, graph in enumerate(dataloader_train):

                # Get target and mask from graph
                train_mask = graph['high'].train_mask
                y = graph["high"].y

                y_out = model(graph)
                loss = loss_fn(y_out) # the loss internally handles the different y_out cases
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                #accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += 1
                
                # Log values to wandb
                loss_meter.update(val=loss.item(), n=y.shape[0])
                
                accelerator.log({
                    'epoch':epoch,
                    'train loss iteration': loss_meter.val,
                    'train loss avg': loss_meter.avg,
                }, step=step)

            end = time.time()

            accelerator.log({
                'epoch':epoch,
                'train loss avg': loss_meter.avg,
                'lr': np.mean(lr_scheduler.get_last_lr())
            }, step=step)

            write_log(
                f"\nEpoch {epoch} completed in {end - start:.4f} seconds." +
                f"Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ", args, accelerator, 'a'
            )
                    
            accelerator.save_state(output_dir=args.output_path+f"checkpoints/checkpoint_{epoch}/", safe_serialization=False)
            torch.save({"epoch": epoch}, args.output_path+f"checkpoints/checkpoint_{epoch}/epoch")

            # VALIDATION
            if dataloader_val is not None:
                model.eval()

                # if epoch%5==0:
                if log_val_plots:
                    y_plot_list = []
                    y_pred_plot_list = []
                    idxs_list = []

                with torch.no_grad():    
                    for graph in dataloader_val:
                        
                        # Get target and mask from graph
                        train_mask = graph['high'].train_mask
                        y = graph["high"].y

                        y_out = model(graph) # mu, phi if tweedie loss
                        loss = loss_fn(y_out)

                        val_loss_meter.update(val=loss.item(), n=y.shape[0])
                        accelerator.log({
                            'epoch':epoch,
                            'val loss iteration': val_loss_meter.val,
                            'val loss avg': val_loss_meter.avg
                        }, step=step)
                        
                        if log_val_plots:
                            y_plot, y_pred_plot = get_final_values(y, y_out, args)
                            # retrieve graphs for individual time instances
                            n_nodes = graph["high"].num_nodes
                            B = y_plot.shape[0] // n_nodes
                            y_plot = y_plot.view(B, n_nodes)
                            y_pred_plot = y_pred_plot.view(B, n_nodes, -1)
                            train_mask = train_mask.view(B, n_nodes)

                            y_pred_plot = torch.atleast_2d(y_pred_plot) # from (N,) to (1,N)
                            y_plot = torch.atleast_2d(y_ploty)
                            idxs = torch.atleast_2d(torch.tensor(graph.idxs, device=accelerator.device))
                            y_pred_plot_list.append(y_pred_plot) # time, nodes
                            y_plot_list.append(y_plot)
                            idxs_list.append(idxs)     

                    ###### PLOTS ######
                    if log_val_plots:
                        y_pred_plot_list_all = accelerator.gather(torch.stack(y_pred_plot_list)).swapaxes(0,1)[:,:val_size] # (nodes, time) (449152, 48, 32)
                        y_plot_all = accelerator.gather(torch.stack(y_plot_list)).swapaxes(0,1)[:,:val_size]
                        idxs_all = accelerator.gather(torch.stack(idxs_list)).squeeze()[:val_size]

                        # Create a few plots to compare
                        create_validation_plots(y_pred_plot_list_all, y_plot_all, idxs_all, times, graph, accelerator, step, epoch, args)
                            
                accelerator.log({
                    'epoch':epoch,
                    'val loss avg': val_loss_meter.avg,
                }, step=step)
                    
            if lr_scheduler is not None:
                lr_scheduler.step()  