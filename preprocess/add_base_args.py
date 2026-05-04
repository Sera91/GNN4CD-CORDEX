def add_base_args(parser):

    #-- paths
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--input_path_target', type=str)
    parser.add_argument('--input_path_predictors', type=str)
    parser.add_argument('--input_path_topo', type=str)
    parser.add_argument('--input_path_mask_sealand', type=str)
    parser.add_argument('--target_file', type=str)
    parser.add_argument('--predictors_file', type=str)
    parser.add_argument('--topo_file', type=str)
    parser.add_argument('--mask_sealand_file', type=str)
    parser.add_argument('--land_use_path', type=str)
    parser.add_argument('--land_use_file', type=str)

    #-- lat lon grid values
    parser.add_argument('--lon_min', type=float)
    parser.add_argument('--lon_max', type=float)
    parser.add_argument('--lat_min', type=float)
    parser.add_argument('--lat_max', type=float)
    parser.add_argument('--lon_grid_radius_high', type=float)
    parser.add_argument('--lat_grid_radius_high', type=float)
    parser.add_argument('--lon_grid_radius_low', type=float, default=0.36)
    parser.add_argument('--lat_grid_radius_low', type=float, default=0.36)

    #-- other
    parser.add_argument('--mask_path', type=str, default=None)
    parser.add_argument('--mask_file', type=str, default=None)
    parser.add_argument('--predictors_dataset', type=str)
    parser.add_argument('--target_dataset', type=str)
    parser.add_argument('--target_type', type=str, default="precipitation")
    parser.add_argument('--target_multiplier', type=float, default=1)

    parser.add_argument('--low_transformed_time_res', type=str, default="1h")
    parser.add_argument('--high_transformed_time_res', type=str, default="1h")

    #-- era5
    parser.add_argument('--input_files_suffix_low', type=str, default='')
    parser.add_argument('--n_levels_low', type=int, help='number of pressure levels considered', default=5)

    return parser