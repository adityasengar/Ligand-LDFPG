# Diffusion Latent Training (EDM)

A PyTorch implementation for training and sampling a latent-space diffusion model using the Elucidated Diffusion Model (EDM) framework. Includes tools for single runs and large-scale hyperparameter sweeps on SLURM-based clusters.

---

## Directory Structure

```
.
├── edm_diff.py             # Main training & sampling script
├── edm_sweep.sh           # Bash script for SLURM sweep
├── latent_reps/
│   └── pooled_embedding.h5 # Example HDF5 dataset (input)
├── checkpoints/            # Model checkpoints (output)
├── generated/              # Generated samples (output)
├── logs/                   # Training logs (output)
```

---

## 1. Environment Setup

```bash
# Load modules (adapt if needed)
module load gcc python cuda

# Activate your Python virtual environment
source ~/venvs/venv2/bin/activate
```

## 2. Single Training Run

Run the model on your data:

```bash
python edm_diff.py \
  --data_file latent_reps/pooled_embedding.h5 \
  --output_model_path checkpoints/edm_latent_model.pth \
  --output_samples_path generated/edm_latents.h5 \
  --log_file logs/edm_latent_training.log \
  --epochs 500 \
  --batch_size 128 \
  --cuda_device 0
```

* See `python edm_diff.py --help` for all options.

## 3. Hyperparameter Sweep (Recommended for Cluster)

Launch the sweep on a SLURM system:

```bash
sbatch sweep_gpu2.sh
```

* Each run is configured with a unique combination of hyperparameters.
* Output:
  * Logs in `logs/sweep_runs/`
  * Checkpoints in `checkpoints/sweep_runs/`
  * Generated samples in `generated/sweep_runs/`

## 4. Input Data

* Expects an HDF5 file with shape `[num_samples, latent_dim]` at the provided `--data_file` and `--data_key`.
* Default key: `pooled_embedding`.

## 5. Outputs

* **Checkpoints:** in `checkpoints/`
* **Samples:** in `generated/`
* **Logs:** in `logs/`

## Tips

* Update file paths in the scripts as needed.
* Tune `SBATCH` settings in the bash script for your cluster.
* Logs will contain all runtime info and errors.

## Citation

If you use or adapt this code, please cite the original repo or contact the author.
