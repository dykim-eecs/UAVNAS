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
#  - Module Name : model_search
#  - Function    : DARTS differentiable architecture search super-network
#
#

"""DARTS super-network: a stack of MixedOp-based cells with learnable architecture alphas."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genotypes import Genotype
from operations import OPS, PRIMITIVES, FactorizedReduce, ReLUConvBN


class MixedOp(nn.Module):
    """Convex combination of every primitive op on a single edge."""

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            # affine=False during search keeps BN parameters from interfering with the alphas
            op = OPS[primitive](C, stride, affine=False)
            self.ops.append(op)

    def forward(self, x, weights):
        # ``weights`` is the per-op softmax distribution for this edge
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class Cell(nn.Module):
    """A single search cell (normal or reduction)."""

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier

        # Preprocess the two previous cell outputs so they share C channels
        if reduction_prev:
            # Previous cell was a reduction; align resolution and channels first
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        # Build a MixedOp for every (intermediate-node, predecessor) edge
        # Node indices: 0,1 are the cell inputs (s0, s1); 2..(1+steps) are intermediate nodes
        self._ops = nn.ModuleList()
        edge_index = 0
        for i in range(self.steps):
            for j in range(i + 2):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)
                edge_index += 1

    def forward(self, s0, s1, weights):
        # s0: output from two cells back, s1: output from the previous cell
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            # Output of node i+2 = weighted sum of every predecessor's MixedOp output
            res = 0
            for j in range(i + 2):
                res = res + self._ops[offset + j](states[j], weights[offset + j])
            states.append(res)
            offset += i + 2
        # Concatenate the last ``multiplier`` intermediate nodes as the cell output
        return torch.cat(states[-self.multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion=None,
                 steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C  # initial channel count
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        # Stem: project the 2-channel I/Q input up to (C * stem_multiplier) channels
        input_channels = 2
        C_cur = C * stem_multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, C_cur, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_cur),
        )

        # Stack of cells; reductions sit at 1/3 and 2/3 depth (DARTS convention)
        self.cells = nn.ModuleList()
        C_prev_prev = C_cur
        C_prev = C_cur
        reduction_prev = False
        reduction_layers = [layers // 3, 2 * layers // 3]
        for i in range(layers):
            if i in reduction_layers:
                C_curr = C * 2
                reduction = True
            else:
                C_curr = C
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            C_prev_prev = C_prev
            C_prev = C_curr * multiplier
            if reduction:
                C = C * 2

        # Global pooling + linear classifier head
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # Architecture parameters (alphas) — one row per edge
        k = 0
        for i in range(self._steps):
            k += i + 2
        num_ops = len(PRIMITIVES)
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    def weights(self):
        # All learnable tensors except the alphas
        return [param for name, param in self.named_parameters() if "alpha" not in name]

    def forward(self, x):
        s0 = self.stem(x)
        s1 = s0
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def genotype(self):
        """Discretize the current alphas into a fixed Genotype."""
        gene_normal = []
        gene_reduce = []
        for alphas, gene in [(self.alphas_normal, gene_normal),
                             (self.alphas_reduce, gene_reduce)]:
            weights = alphas.data.cpu().numpy()
            n = 0
            for i in range(self._steps):
                offset = i + 2
                edge_candidates = []
                for j in range(offset):
                    ops_weights = weights[n + j]
                    # PRIMITIVES[0] == 'none' is excluded from selection
                    idx_best = np.argmax(ops_weights[1:]) + 1
                    op_name = PRIMITIVES[idx_best]
                    edge_candidates.append((op_name, j, ops_weights[idx_best]))
                # Keep the two strongest edges per intermediate node
                edge_candidates.sort(key=lambda x: x[2], reverse=True)
                top2 = edge_candidates[:2]
                gene.extend([(op, idx) for op, idx, _ in top2])
                n += offset
        normal_concat = list(range(2, 2 + self._steps))
        reduce_concat = list(range(2, 2 + self._steps))
        return Genotype(normal=gene_normal, normal_concat=normal_concat,
                        reduce=gene_reduce, reduce_concat=reduce_concat)
