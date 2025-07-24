#!/usr/bin/env python3
import os
import glob
import re
import matplotlib.pyplot as plt

# Path to your logs directory
LOG_DIR = '/scratch/asengar/long_sim/apo_d2_inv_start/run6/full_code/blind_res_CA_off_zref/runs/test_new/run_HNO16/diffusion/normal/run_h50_w2_d12_edm/logs'

def extract_runs(path, max_runs=5):
    """
    Read a sweep file, split into runs, and return up to max_runs
    of (run_id, list of (epoch, loss)) tuples.
    """
    with open(path, 'r') as f:
        text = f.read()

    # split file on run headers like '--- RUN 65/128 ---'
    parts = re.split(r'--- RUN (\d+)/\d+ ---', text)
    chunks = list(zip(parts[1::2], parts[2::2]))

    runs = []
    for run_id_str, block in chunks:
        run_id = int(run_id_str)
        # find all epoch-loss pairs
        losses = [
            (int(m.group(1)), float(m.group(2)))
            for m in re.finditer(r'\[INFO\] Epoch (\d+)/\d+ \| Avg Loss: ([0-9.]+)', block)
        ]
        if losses:
            runs.append((run_id, losses))
        if len(runs) >= max_runs:
            break

    return runs

def plot_file(path):
    """
    Generate an overlay plot for up to five runs in a single file.
    """
    runs = extract_runs(path, max_runs=5)
    if not runs:
        print(f"No runs found in {os.path.basename(path)}")
        return

    plt.figure()
    for run_id, losses in runs:
        epochs, vals = zip(*losses)
        plt.plot(epochs, vals, label=f'Run {run_id}')
    plt.title(os.path.basename(path))
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    pattern = os.path.join(LOG_DIR, 'sweep_gpu_*.out')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No sweep files matching {pattern}")
        return

    for path in files:
        plot_file(path)

if __name__ == '__main__':
    main()
