#!/usr/bin/env python3
import os
import glob
import re
import matplotlib.pyplot as plt
import pandas as pd

# folder with individual run logs
LOG_DIR = '/scratch/asengar/long_sim/apo_d2_inv_start/run6/full_code/blind_res_CA_off_zref/runs/test_new/run_HNO16/diffusion/normal/run_h50_w2_d12/logs/sweep_runs'

def extract_losses(log_path):
    """
    Read one .log file and return a list of (epoch, loss) tuples.
    """
    pattern = re.compile(r'Epoch\s+(\d+)/\d+\s+\|\s+Avg Loss:\s+([0-9.]+)')
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                data.append((int(m.group(1)), float(m.group(2))))
    return data

def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def main():
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, '*.log')))
    if not log_files:
        print(f"No .log files in {LOG_DIR}")
        return

    summary = []

    # plot in batches of 10
    for batch_idx, batch in enumerate(chunked(log_files, 10), start=1):
        plt.figure(figsize=(10, 6))
        for log_path in batch:
            runs = extract_losses(log_path)
            if not runs:
                continue

            epochs, losses = zip(*runs)
            name = os.path.splitext(os.path.basename(log_path))[0]
            plt.plot(epochs, losses, label=name)

            # record lowest loss for this run
            best_epoch, best_loss = min(runs, key=lambda x: x[1])
            summary.append({
                'run': name,
                'best_epoch': best_epoch,
                'best_loss': best_loss
            })

        plt.xlabel('Epoch')
        plt.ylabel('Average Loss (log scale)')
        plt.yscale('log')
        plt.ylim(bottom=0.005)  # ensure y-axis goes down to 0.005
        start = (batch_idx - 1) * 10 + 1
        end = min(batch_idx * 10, len(log_files))
        plt.title(f'Runs {start}–{end}')
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        plt.show()

    # print summary table
    if summary:
        df = pd.DataFrame(summary)
        df = df.sort_values('run').reset_index(drop=True)
        print("\nLowest loss per simulation:\n")
        print(df.to_string(index=False))

if __name__ == '__main__':
    main()
