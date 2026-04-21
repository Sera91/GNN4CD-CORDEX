import pickle
import torch
import numpy

def create_validation_plots(
    y_pred_plot, y_plot, t, times, graph, accelerator, step, epoch, args, binmax=350,
    metadata_file_path="/leonardo_work/ICT26_ESP/vblasone/GNN4CD-CORDEXML/utils/CORDEXML_plot_params.json"):

    with open(metadata_file_path) as f:
        meta = json.load(f)

    meta = convert_dict(meta)
    target_type = args.target_type
    
    lon = graph['high'].lon.cpu().numpy()
    lat = graph['high'].lat.cpu().numpy()

    # convert to cpu and numpy
    _, indices = torch.sort(t)
    indices = indices.cpu().numpy()
    times = times[indices]

    y_pred_pdf = y_pred_plot.flatten()
    y_pdf = y_plot.flatten()
    binmin = min(np.floor(np.min(y_pred_plot)), np.floor(np.min(y_plot))) - 5
    binmax = max(np.ceil(np.max(y_pred_plot)), np.ceil(np.min(y_plot))) + 5
    bins = np.arange(binmin,binmax,1).astype(np.float32)

    cmap_dict = get_cmap_dict()
    bounds_avg = [0, 1, 1.5, 2, 4, 6, 8, 10, 12] #, 15, 20] #, 25, 30, 35]
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds_avg, ncolors=256)

    gnn4cd_avg = np.nanmean(y_pred_plot, axis=-1)
    target_avg = np.nanmean(y_plot, axis=-1)

    bias =  gnn4cd_avg - target_avg

    if meta[target_type]["cmap"] == "cmap_dict['avg']['cmap']":
        cmap = cmap_dict['avg']['cmap']
    else:
        cmap = meta[target_type]["cmap"]

    fig_avg = plot_maps(
        [lon, lon],
        [lat, lat],
        [gnn4cd_avg, target_avg],
        aggr=None,
        s=meta[target_type]["s"],
        legend_title=meta[target_type]["map_unit"],
        cmap=cmap,
        sub_titles=["GNN4CD", "TARGET"],
        x_size=meta["general"]["figsize"][0],
        y_size=meta["general"]["figsize"][1],
        font_size_title=meta["general"]["fontsize_title"],
        font_size=meta["general"]["fontsize"],
        cbar_title_size=meta["general"]["fontsize_cbar_title"],
        pr_max=meta[target_type]["vmax"],
        pr_min=meta[target_type]["vmin"],
        cbar_pad=20,
        suptitle_y=0.87,
        suptitle_x=0.72,
        show_ticks=False,
        plot_func="scatter",
        xlim=meta["general"]["xlim"],
        ylim=meta["general"]["ylim"],
        proj=ccrs.PlateCarree(),
        cbar_ax_lim=[0.93,0.23,0.015,0.55]
    )

    fig_bias = plot_maps(
        lon,
        lat,
        bias,
        aggr=None,
        s=meta[target_type]["s"],
        legend_title=meta[target_type]["map_unit"],
        cmap="BrBG",
        sub_titles=["GNN4CD - TARGET"],
        x_size=meta["general"]["figsize"][0]/2,
        y_size=meta["general"]["figsize"][1],
        font_size_title=25,
        font_size=20,
        cbar_title_size=20,
        pr_max=meta[target_type]["vmax_bias"],
        pr_min=meta[target_type]["vmin_bias"],
        cbar_pad=20,
        suptitle_y=0.87,
        suptitle_x=0.72,
        show_ticks=False,
        plot_func="scatter",
        xlim=meta["general"]["xlim"],
        ylim=meta["general"]["ylim"],
        proj=ccrs.PlateCarree(),
        cbar_ax_lim=[0.93,0.23,0.015,0.55]
    )

    hist_vals, bins = np.histogram(y_pred_pdf, bins=bins, density=False)
    bins_mid = (bins[:-1] + bins[1:]) / 2
    Ntot = np.nansum(hist_vals)
    hist_vals_target, bins_target = np.histogram(y_pdf, bins=bins, density=False)
    bins_target_mid = (bins_target[:-1] + bins_target[1:]) / 2
    Ntot_target = np.nansum(hist_vals_target)

    if meta[target_type]["xlim_pdf"] is None:
        meta[target_type]["xlim_pdf"] = [0.2, bins.max()+10]

    fig_pdf = plot_pdf(
        bin_list=[bins_target_mid, bins_mid],
        hist_list=[hist_vals_target/Ntot_target, hist_vals/Ntot],
        label_list=["TARGET", "GNN4CD"],
        xlabel=meta[target_type]["pdf_unit"],
        color_list=["black", "darkorange"],
        tail_lim=meta[target_type]["tail_lim"],
        ylim=meta[target_type]["ylim_pdf"],
        title=meta[target_type]["pdf_title"],
        xlim=meta[target_type]["xlim_pdf"],
        plot_func=meta[target_type]["plot_func_pdf"],
        fontsize=20,
        suptitle="",
        tail_ylim=meta[target_type]["tail_ylim"],
        log_xy=meta[target_type]["log_xy"],
        tail_zoom=meta[target_type]["tail_zoom"],
        legend_outside=meta[target_type]["legend_outside"]
    )

    accelerator.log({
        meta[target_type]["map_title"]: [wandb.Image(fig_avg)],
        "bias":  [wandb.Image(fig_bias)],
        "pdf": [wandb.Image(fig_pdf)],
        }, step=step)
    
    plt.close(fig_avg)
    plt.close(fig_bias)
    plt.close(fig_pdf)

    if epoch == (args.epochs-1): # last epoch
        data = HeteroData()
        if args.target_type == "precipitation":
            data.pr_gnn4cd = y_pred_plot
        elif args.target_type == "temperature":
            data.tasmax_gnn4cd = y_pred_plot
        
        data.target = y_plot

        data.times = times
        data.times_target = times
        data["high"].lat = lat
        data["high"].lon = lon

        with open(args.output_path + f"output_graph_{args.validation_year}.pkl", 'wb') as f:
            pickle.dump(data, f)
