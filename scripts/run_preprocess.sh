#!/bin/bash
source $1

mkdir -p ${OUTPUT_PATH}
LOG_PATH=${OUTPUT_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --time ${TIME}       # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1   # 8 tasks out of 128
#SBATCH --job-name=${JOB_NAME}
# SBATCH --mail-type=FAIL,END
# SBATCH --mail-user=${MAIL}
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

source ${SOURCE_PATH}

module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

conda activate ${ENV_PATH}

cd ${MAIN_PATH}
export PYTHONPATH=$(pwd):$PYTHONPATH

python -m preprocess.preprocess \
--input_path_predictors=${INPUT_PATH_PREDICTORS} \
--input_path_target=${INPUT_PATH_TARGET} \
--input_path_topo=${INPUT_PATH_TOPO} \
--input_path_mask_sealand=${INPUT_PATH_MASK_SEALAND} \
--target_file=${TARGET_FILE} \
--predictors_file=${PREDICTORS_FILE} \
--mask_sealand_file=${MASK_SEALAND_FILE} \
--topo_file=${TOPO_FILE} \
--output_path=${OUTPUT_PATH} \
--log_file=${LOG_FILE} \
--lon_grid_radius_high=${LON_GRID_RADIUS_HIGH} \
--lat_grid_radius_high=${LAT_GRID_RADIUS_HIGH} \
--mask_path=${MASK_PATH} \
--mask_file=${MASK_FILE} \
--land_use_path=${LAND_USE_PATH} \
--land_use_file=${LAND_USE_FILE} \
--target_type=${TARGET_TYPE} \
--target_multiplier=${TARGET_MULTIPLIER} \
--low_transformed_time_res=${LOW_TRANSFORMED_TIME_RES} \
--high_transformed_time_res=${HIGH_TRANSFORMED_TIME_RES}

EOT


