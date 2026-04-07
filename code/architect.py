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
#  - Module Name : architect
#  - Function    : Architecture parameter optimizer for DARTS search
#
#

"""Optimizer wrapper that updates the architecture parameters (alphas) for DARTS."""

import torch


class Architect:
    def __init__(self, model, criterion, arch_lr, arch_weight_decay):
        """Manage updates to the architecture parameters (alphas) of ``model``."""
        self.model = model
        self.criterion = criterion
        # Adam optimizer dedicated to the architecture parameters only
        self.optimizer = torch.optim.Adam(
            model.arch_parameters(),
            lr=arch_lr,
            betas=(0.5, 0.999),
            weight_decay=arch_weight_decay,
        )

    def step(self, input_train, target_train, input_valid, target_valid, unrolled=False):
        """Run a single update of the architecture parameters.

        ``unrolled=True`` would enable second-order (unrolled) optimization, but this
        implementation uses the simpler first-order approximation.
        """
        # Reset gradients on the architecture optimizer
        self.optimizer.zero_grad()
        # Also clear model gradients to avoid contaminating the weight optimizer state
        self.model.zero_grad()
        # Forward + loss on the validation batch
        was_training = self.model.training
        self.model.eval()
        logits_valid = self.model(input_valid)
        loss_valid = self.criterion(logits_valid, target_valid)
        # Backprop computes grads for both weights and alphas; only the alpha grads are used
        loss_valid.backward()
        # Apply the alpha update
        self.optimizer.step()
        # Restore the previous training mode
        if was_training:
            self.model.train()
