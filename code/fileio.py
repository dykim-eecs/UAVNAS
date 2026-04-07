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
#  - Module Name : fileio
#  - Function    : IQ waveform dataset loader for drone RF signals (.fc32 format)
#
#

"""PyTorch Dataset that loads drone IQ waveforms stored as raw float32 (.fc32) files."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class WaveformDataset(Dataset):
    """Dataset of drone IQ waveforms.

    Each ``.fc32`` file is interpreted as interleaved float32 I/Q samples and reshaped
    into a ``(2, 32, 3072)`` tensor (2 channels = I/Q, 32x3072 = the time/frequency grid
    consumed by the network). Per-sample normalization is applied for training stability.
    """

    def __init__(self, root_dir):
        self.files = []
        self.labels = []

        # Class subdirectories are named '00', '01', ..., '12'
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = int(class_name)
            file_list = [
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.endswith('.fc32')
            ]
            file_list.sort()
            self.files.extend(file_list)
            self.labels.extend([label] * len(file_list))

        assert len(self.files) == len(self.labels), "File/label count mismatch"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        # Load the .fc32 file as raw float32
        data = np.fromfile(file_path, dtype=np.float32)

        try:
            iq_pairs = data.reshape(-1, 2)
        except ValueError:
            raise ValueError(f"File {file_path} has an unexpected size.")

        iq_pairs = iq_pairs[:98304, :]
        iq_tensor = torch.from_numpy(iq_pairs).T.view(2, 32, 3072)
        # Per-sample normalization
        iq_tensor = (iq_tensor - iq_tensor.mean()) / (iq_tensor.std() + 1e-6)
        return iq_tensor, label
