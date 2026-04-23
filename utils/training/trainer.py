import torch
import numpy as np
import pickle
import time
import wandb
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
import json

from utils.metrics import AverageMeter
from utils.helpers import write_log
from utils.plotting import create_validation_plots
from utils.extractors import extract_prediction
from utils.helpers import convert_dict
from utils.predictand_transforms import inverse_transform_predictand

#-----------------------------------------------------
#---------------------- TRAIN ------------------------
#-----------------------------------------------------

class Trainer(object):

    def __init__(self):
        super(Trainer, self).__init__()

    def train(
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
                y_trans = graph["high"].y

                y_out_trans = model(graph)
                loss = loss_fn(y_out_trans, y_trans) # the loss internally handles the different y_out cases
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                #accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += 1
                
                # Log values to wandb
                loss_meter.update(val=loss.item(), n=y_trans.shape[0])
                
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
                if args.make_val_plots and epoch % args.val_plot_frequency==0:
                    y_list = []
                    ypred_list = []
                    idxs_list = []

                with torch.no_grad():    
                    for graph in dataloader_val:
                        
                        # Get target and mask from graph
                        train_mask = graph['high'].train_mask
                        y_trans = graph["high"].y

                        y_out_trans = model(graph)
                        loss = loss_fn(y_out_trans, y_trans)

                        val_loss_meter.update(val=loss.item(), n=y_trans.shape[0])
                        accelerator.log({
                            'epoch':epoch,
                            'val loss iteration': val_loss_meter.val,
                            'val loss avg': val_loss_meter.avg
                        }, step=step)
                        
                        if args.make_val_plots:
                            y_pred_trans = extract_prediction(y_out_trans, args.loss_fn)
                            stats = np.load(args.output_path+"predictand_stats.npz", allow_pickle=True)
                            y = inverse_transform_predictand(y_trans, stats)
                            y_pred = inverse_transform_predictand(y_pred_trans, stats)

                            # retrieve graphs for individual time instances
                            n_nodes = graph["high"].num_nodes
                            B = y.shape[0] // n_nodes
                            y = y.view(B, n_nodes)
                            y_pred = y_pred.view(B, n_nodes, -1)
                            train_mask = train_mask.view(B, n_nodes)

                            y_pred = torch.atleast_2d(y_pred) # from (N,) to (1,N)
                            y = torch.atleast_2d(y)
                            idxs = torch.atleast_2d(torch.tensor(graph.idxs, device=accelerator.device))
                            y_pred_list.append(y_pred) # time, nodes
                            y_list.append(y)
                            idxs_list.append(idxs)     

                    ###### PLOTS ######
                    if args.make_val_plots:
                        y_pred_all = accelerator.gather(torch.stack(y_pred_list)).swapaxes(0,1)[:,:val_size] # (nodes, time) (449152, 48, 32)
                        y_all = accelerator.gather(torch.stack(y_list)).swapaxes(0,1)[:,:val_size]
                        idxs_all = accelerator.gather(torch.stack(idxs_list)).squeeze()[:val_size]

                        metadata_file_path="/leonardo_work/ICT26_ESP/vblasone/GNN4CD-CORDEXML/utils/CORDEXML_plot_params.json"
                        with open(metadata_file_path) as f:
                            meta = json.load(f)

                        meta = convert_dict(meta)
                        target_type = args.target_type
                        
                        lon = graph['high'].lon.cpu().numpy()
                        lat = graph['high'].lat.cpu().numpy()

                        # convert to cpu and numpy
                        _, indices = torch.sort(idxs_all)
                        indices = indices.cpu().numpy()
                        times = times[indices]

                        # Create a few plots to compare
                        fig_avg, fig_bias, fig_pdf = create_validation_plots(
                            y_pred_all[indices], # to ensure they are sorted correctly
                            y_all[indices],
                            lon,
                            lat,
                            args.target_type,
                            meta
                        )

                        accelerator.log({
                            "average": [wandb.Image(fig_avg)],
                            "bias":  [wandb.Image(fig_bias)],
                            "pdf": [wandb.Image(fig_pdf)],
                            }, step=step)

                        plt.close(fig_avg)
                        plt.close(fig_bias)
                        plt.close(fig_pdf)

                        if epoch == (args.epochs-1): # last epoch
                            data = HeteroData()
                            if args.target_type == "precipitation":
                                data.pr_gnn4cd = y_pred
                            elif args.target_type == "temperature":
                                data.tasmax_gnn4cd = y_pred
                            
                            data.target = y

                            data.times = times
                            data.times_target = times
                            data["high"].lat = lat
                            data["high"].lon = lon

                            with open(args.output_path + f"output_graph_{args.validation_year}.pkl", 'wb') as f:
                                pickle.dump(data, f)
                            
                accelerator.log({
                    'epoch':epoch,
                    'val loss avg': val_loss_meter.avg,
                }, step=step)
                    
            if lr_scheduler is not None:
                lr_scheduler.step()  