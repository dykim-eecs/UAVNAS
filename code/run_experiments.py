#!*****************************************************************************************
#! [UAV-NAS Project] CONFIDENTIAL & PROPRIETARY                                            *
#! Copyright (c) 2026 Doh-Yon Kim (First Author), All Rights Reserved.                     *
#!                                                                                         *
#! NOTICE: This source code is associated with the following academic publication:         *
#! "UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search"                   *
#! Accepted for publication in IEEE Transactions on Industrial Informatics (TII).          *
#!                                                                                         *
#! All information contained herein is, and remains the property of the authors.           *
#! The intellectual and technical concepts contained herein are protected by copyright     *
#! law and international treaties. Unauthorized copying, modification, or distribution     *
#! of this file, via any medium, is strictly prohibited.                                   *
#!                                                                                         *
#!*****************************************************************************************
#
#* [ PAPER METADATA ]
#  - Title   : UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search
#  - Journal : IEEE Transactions on Industrial Informatics (TII)
#
#* [ AUTHOR INFORMATION ]
#  - First Author : Kim, Doh-Yon
#
#* [ FILE DESCRIPTION ]
#  - Module Name : run_experiments
#  - Function    : Hyperparameter sweep runner over (init_channels, layers) combinations
#
#

"""Sweep over (init_channels, layers) by repeatedly invoking ``train_search.py``.

Each run's stdout is parsed for per-epoch metrics, which are streamed to a timestamped
CSV so partial results survive a crash.
"""

import os
import re
import subprocess
import time

import pandas as pd

# ===== Sweep configuration =====
init_channels_values = [1, 2, 4, 8, 16]
layers_values = [1, 2, 4, 8, 16]
epochs = 10
train_dir = '/home/dykim/drone/dronedetect/train'

# Timestamped CSV path so concurrent runs do not collide
timestamp = time.strftime("%Y%m%d_%H%M%S")
csv_path = f"full_experiment_results_{timestamp}.csv"
all_results = []

# Initialize the CSV with the header row
if not os.path.exists(csv_path):
    pd.DataFrame(columns=[
        'init_channels', 'layers', 'epoch',
        'train_loss', 'val_loss', 'accuracy', 'num_parameters',
    ]).to_csv(csv_path, index=False)

print("Starting experiments. Each run is capped at the configured epoch budget; "
      "results are appended to the CSV as they arrive...")

for init_channels in init_channels_values:
    for layers in layers_values:
        print(f"\n--- Starting run: init_channels={init_channels}, layers={layers} ---")
        command = [
            "python3", "train_search.py",
            "--epochs", str(epochs),
            "--init_channels", str(init_channels),
            "--layers", str(layers),
            "--train_dir", train_dir,
            "--patience", "5",  # Early stopping patience for the sweep
        ]

        num_parameters = None

        try:
            # Capture stdout from the train_search.py subprocess
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output_lines = result.stdout.splitlines()

            for line in output_lines:
                # Total parameter count line
                param_match = re.search(r'Total model parameters: (\d+)', line)
                if param_match:
                    num_parameters = int(param_match.group(1))

                # Per-epoch metrics line: Train Loss, Val Loss, Val Acc
                epoch_match = re.search(
                    r'\[Epoch (\d+)/(\d+)\] Train Loss: (\d+\.\d+), .* '
                    r'Val Loss: (\d+\.\d+), Val Acc: (\d+\.\d+)%',
                    line,
                )
                if epoch_match:
                    epoch_num = int(epoch_match.group(1))
                    train_loss = float(epoch_match.group(3))
                    val_loss = float(epoch_match.group(4))
                    val_acc = float(epoch_match.group(5))

                    all_results.append({
                        'init_channels': init_channels,
                        'layers': layers,
                        'epoch': epoch_num,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'accuracy': val_acc,
                        'num_parameters': num_parameters,
                    })

                    # Stream the latest row to disk immediately
                    pd.DataFrame([all_results[-1]]).to_csv(
                        csv_path, mode='a', header=False, index=False)

            if not any(r['init_channels'] == init_channels and r['layers'] == layers
                       for r in all_results):
                raise ValueError(
                    f"No results parsed for init_channels={init_channels}, layers={layers}.")

            print(f"Run init_channels={init_channels}, layers={layers} complete.")

        except subprocess.CalledProcessError as e:
            print(f"Error: run init_channels={init_channels}, layers={layers} failed")
            print(f"stderr: {e.stderr}")
            continue
        except ValueError as e:
            print(e)
            continue

# ===== Final aggregation =====
if all_results:
    df_full_results = pd.DataFrame(all_results)

    # Pick each run's last reported epoch (handles early-stopped runs uniformly)
    final_epoch_results = df_full_results.groupby(['init_channels', 'layers']).apply(
        lambda x: x[x['epoch'] == x['epoch'].max()]
    ).reset_index(drop=True)

    # Pivot into a (init_channels x layers) table
    table = final_epoch_results.pivot_table(
        index='init_channels',
        columns='layers',
        values=['train_loss', 'val_loss', 'accuracy', 'num_parameters'],
        aggfunc='first',
    )

    print("\n--- Final-epoch results (Train Loss / Val Loss / Accuracy / Num Parameters) ---")
    print(table)

    print(f"\nAll experiment results saved to '{csv_path}'.")
else:
    print("No experiment results were produced. Check the script's stderr output.")
