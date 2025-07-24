#!/bin/bash
#SBATCH -J diff_gpu2
#SBATCH --output=logs/sweep_gpu_2.out
#SBATCH --time=48:00:00
#SBATCH --partition=h100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64000

# --- Setup Environment ---
echo "Loading modules..."
module load gcc python cuda
source ~/venvs/venv2/bin/activate
echo "Environment activated."

# --- Define Hyperparameter Space ---
LRS=(1e-4 2e-4)
STEPS=(20 40 60 80)
PMEANS=(-1.2 -1.5)
PSTDS=(1.2 1.5)
HDIMS=(512 1024)
NLAYERS=(4 6)

# --- Create Directories ---
LOG_DIR="logs/sweep_runs"
MODEL_DIR="checkpoints/sweep_runs"
SAMPLES_DIR="generated/sweep_runs"
mkdir -p $LOG_DIR $MODEL_DIR $SAMPLES_DIR

# --- Main Loop ---
COUNT=0
echo "Starting hyperparameter sweep for GPU 2 (runs 65-96)..."

for LR in "${LRS[@]}"; do
for STEP in "${STEPS[@]}"; do
for PMEAN in "${PMEANS[@]}"; do
for PSTD in "${PSTDS[@]}"; do
for HDIM in "${HDIMS[@]}"; do
for NLAYER in "${NLAYERS[@]}"; do

    COUNT=$((COUNT + 1))
    # This script handles runs 65 to 96
    if [ $COUNT -le 64 ] || [ $COUNT -gt 96 ]; then
        continue
    fi

    echo "--- RUN ${COUNT}/128 ---"
    
    # --- Define unique filenames for this run ---
    FILENAME="lr${LR}_steps${STEP}_pmean${PMEAN}_pstd${PSTD}_hdim${HDIM}_nlayers${NLAYER}"
    LOG_FILE="${LOG_DIR}/${FILENAME}.log"
    MODEL_PATH="${MODEL_DIR}/${FILENAME}.pth"
    SAMPLES_PATH="${SAMPLES_DIR}/${FILENAME}.h5"

    echo "Parameters:"
    echo "  LR: ${LR}, Steps: ${STEP}, P_mean: ${PMEAN}, P_std: ${PSTD}"
    echo "  Hidden Dim: ${HDIM}, Num Layers: ${NLAYER}"
    echo "  Log File: ${LOG_FILE}"
    
    # --- Execute Python Script ---
    python edm_diff.py \
        --data_file latent_reps/pooled_embedding.h5 \
        --lr $LR \
        --sampler_steps $STEP \
        --P_mean $PMEAN \
        --P_std $PSTD \
        --hidden_dim $HDIM \
        --num_layers $NLAYER \
        --log_file $LOG_FILE \
        --output_model_path $MODEL_PATH \
        --output_samples_path $SAMPLES_PATH \
        --epochs 2000 \
        --batch_size 128 \
        --cuda_device 0

    echo "--- Finished Run ${COUNT} ---"

done; done; done; done; done; done

echo "GPU 2 sweep complete."
deactivate



