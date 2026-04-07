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
#  - Module Name : utils
#  - Function    : Training utilities (AverageMeter, top-k accuracy, EarlyStopping)
#
#

"""Shared training utilities used by both ``train_search.py`` and ``infer.py``."""

import torch


class AverageMeter:
    """Track the most recent value and the running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy for the given outputs/targets (defaults to top-1)."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # Indices of the top-k highest predicted classes
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / batch_size * 100.0).item())
        return res


class EarlyStopping:
    """Stop training when validation loss stops improving.

    The class also caches the best model state and (optionally) the best genotype so
    callers can restore them after the loop ends.

    Parameters
    ----------
    patience : int
        Number of consecutive non-improving epochs allowed before stopping.
    delta : float
        Minimum drop in validation loss that counts as an improvement.
    save_genotype : bool
        When True, ``model.genotype()`` is captured alongside the model state. Used by
        ``train_search.py``; ``infer.py`` keeps it False because the fixed-architecture
        network has no genotype.
    """

    def __init__(self, patience=10, delta=0.0, save_genotype=False):
        self.patience = patience
        self.delta = delta
        self.save_genotype = save_genotype
        self.best_score = None
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_genotype = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss  # Higher score == lower loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.best_val_loss = val_loss
            self.best_model_state = model.state_dict()
            if self.save_genotype and hasattr(model, 'genotype'):
                self.best_genotype = model.genotype()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
