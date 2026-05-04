#!/bin/bash
source $1
mkdir -p ${LOG_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --qos=${QOS} # boost_qos_dbg
#SBATCH --time=${TIME}       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1 # out of 128
#SBATCH --gres=gpu:${N_GPU}       # 1 gpus per node out of 4
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
# SBATCH --mail-user=${MAIL}
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

source ${SOURCE_PATH}

#source ~/anaconda/etc/profile.d/conda.sh
conda activate ${ENV_PATH}

cd ${MAIN_PATH}
export PYTHONPATH=$(pwd):$PYTHONPATH
export CARTOPY_DATA_DIR=${CARTOPY_DATA_DIR}

## Testing
accelerate launch --config_file ${ACCELERATE_CONFIG_PATH} -m predict.predict \
${USE_ACCELERATE} \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH} \
--log_file=${LOG_FILE} \
--graph_file=${GRAPH_FILE} \
--dataset_name=${DATASET_NAME} \
--output_file=${OUTPUT_FILE} \
--test_year_start=${TEST_YEAR_START} \
--test_month_start=${TEST_MONTH_START} \
--test_day_start=${TEST_DAY_START} \
--test_year_end=${TEST_YEAR_END} \
--test_month_end=${TEST_MONTH_END} \
--test_day_end=${TEST_DAY_END} \
--test_years=${TEST_YEARS} \
--batch_size=1 \
--model_name=${MODEL_NAME} \
--seed=${SEED} \
--train_path=${TRAIN_PATH} \
--epoch=${EPOCH} \
--checkpoint=${CHECKPOINT} \
--target_type=${TARGET_TYPE} \
--target_file=${TARGET_FILE} \
--low_input_file=${LOW_INPUT_FILE} \
--orog_file=${OROG_FILE} \
--mask_sealand_file=${MASK_SEALAND_FILE} \
--coords_ij_file=${COORDS_IJ_FILE} \
--metadata_file=${METADATA_FILE} \
--loss_name=${LOSS_NAME} \
--history_length=${HISTORY_LENGTH} \
--threshold=${THRESHOLD}

python ./utils/plotting/plot_report.py \
--input_path=${OUTPUT_PATH} \
--plot_path=${OUTPUT_PATH} \
--val_file="${OUTPUT_FILE}" \
--var=${VAR} \
--experiment="ESD_pseudo_reality" \
--val_year=${TEST_YEAR_START} \
--domain=${DOMAIN} \
--config_file=${CONFIG_FILE_VAL_REPORT}
EOT
