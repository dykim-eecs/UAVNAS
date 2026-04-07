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
#  - Module Name : infer
#  - Function    : Fixed-architecture training, evaluation, and ONNX export pipeline
#
#

"""Train and evaluate the fixed (post-search) network and export it to ONNX."""

import argparse
import csv
import pickle
import subprocess
import sys

import torch
import torch.nn as nn

from fileio import WaveformDataset
from genotypes import NASGenotype
from operations import OPS, FactorizedReduce, ReLUConvBN
from utils import EarlyStopping


class NetworkFixed(nn.Module):
    def __init__(self, genotype, C, num_classes, layers,
                 steps=4, multiplier=4, stem_multiplier=3):
        super(NetworkFixed, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        # Two-channel input: I and Q components of the float32 IQ stream
        input_channels = 2
        C_cur = C * stem_multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, C_cur, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_cur),
        )
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
            cell_genotype = genotype.reduce if reduction else genotype.normal
            cell = Cell(cell_genotype, steps, multiplier, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            C_prev_prev = C_prev
            C_prev = C_curr * multiplier
            if reduction:
                C = C * 2
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = s0
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class Cell(nn.Module):
    def __init__(self, genotype, steps, multiplier, C_prev_prev, C_prev, C,
                 reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)
        self._steps = steps
        self._multiplier = multiplier
        self.ops_by_node = nn.ModuleList()
        self.indices_by_node = []
        for i in range(steps):
            op_name1, inp1 = genotype[2 * i]
            op_name2, inp2 = genotype[2 * i + 1]
            stride1 = 2 if reduction and inp1 < 2 else 1
            stride2 = 2 if reduction and inp2 < 2 else 1
            op1 = OPS[op_name1](C, stride1, affine=True)
            op2 = OPS[op_name2](C, stride2, affine=True)
            self.ops_by_node.append(nn.ModuleList([op1, op2]))
            self.indices_by_node.append((inp1, inp2))

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for ops, (inp1, inp2) in zip(self.ops_by_node, self.indices_by_node):
            h1 = ops[0](states[inp1])
            h2 = ops[1](states[inp2])
            s = h1 + h2
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


def main():
    parser = argparse.ArgumentParser("DRONE-IQ DARTS Inference")
    parser.add_argument('--train_dir', type=str,
                        default='/home/dykim/drone/dronedetect/train')
    parser.add_argument('--test_dir', type=str,
                        default='/home/dykim/drone/dronedetect/test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init_channels', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--classes', type=int, default=13)
    parser.add_argument('--learning_rate', type=float, default=0.025)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--delta', type=float, default=0.001,
                        help='Early stopping delta (minimum loss improvement)')
    parser.add_argument('--save_checkpoint', type=str, default='best_model.pth',
                        help='Path to save the best model checkpoint')
    parser.add_argument('--run_comment3_eval', action='store_true',
                        help='Run the comment3 robustness evaluation after training')
    parser.add_argument('--comment3_script', type=str,
                        default='comment3/eval_robust.py',
                        help='Path to the comment3 robustness evaluation script')
    parser.add_argument('--comment3_out_dir', type=str,
                        default='comment3/results',
                        help='Output directory for comment3 robustness results')
    parser.add_argument('--comment3_workers', type=int, default=0,
                        help='DataLoader workers for the comment3 robustness eval')
    parser.add_argument('--genotype_file', type=str, default=None,
                        help='Path to a pickle genotype that overrides genotypes.NASGenotype')
    parser.add_argument('--onnx_out', type=str, default='model.onnx',
                        help='Output path for the exported ONNX model')
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device('cuda')

    # ===== Data =====
    train_data = WaveformDataset(args.train_dir)
    test_data = WaveformDataset(args.test_dir)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                             shuffle=False, num_workers=8)

    # ===== Model =====
    if args.genotype_file is not None:
        with open(args.genotype_file, 'rb') as f:
            genotype = pickle.load(f)
    else:
        genotype = NASGenotype
    model = NetworkFixed(genotype, C=args.init_channels,
                         num_classes=args.classes, layers=args.layers)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)

    log_data = []

    # ===== Training loop =====
    for epoch in range(args.epochs):
        # ----- Training -----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)
        train_loss_avg = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total

        # ----- Validation -----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
        val_loss_avg = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss_avg:.3f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss_avg:.3f}, Val Acc: {val_acc:.2f}%")

        log_data.append([epoch + 1, train_loss_avg, train_acc, val_loss_avg, val_acc])

        early_stopping(val_loss_avg, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model_state)
            print(f"Best Validation Loss: {early_stopping.best_val_loss:.3f}")
            break

        scheduler.step()

    # Restore best weights and persist the checkpoint for downstream evaluation
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    torch.save(model.state_dict(), args.save_checkpoint)
    print(f"Best checkpoint saved to '{args.save_checkpoint}'")

    # CSV log (only the epochs that actually ran)
    with open("train_val_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(log_data)

    # ===== Final test evaluation (val_loader doubles as the test loader) =====
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(targets).sum().item()
            val_total += targets.size(0)
    final_test_acc = 100.0 * val_correct / val_total
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")

    # ===== ONNX export of the best model =====
    dummy_input = torch.randn(1, 2, 32, 3072, device=device)
    torch.onnx.export(model, dummy_input, args.onnx_out, opset_version=11)
    print(f"ONNX model saved to '{args.onnx_out}'.")

    # Optional: launch the comment3 robustness evaluation script
    if args.run_comment3_eval:
        eval_cmd = [
            sys.executable, args.comment3_script,
            '--test_dir', args.test_dir,
            '--checkpoint', args.save_checkpoint,
            '--batch_size', str(args.batch_size),
            '--classes', str(args.classes),
            '--init_channels', str(args.init_channels),
            '--layers', str(args.layers),
            '--workers', str(args.comment3_workers),
            '--out_dir', args.comment3_out_dir,
        ]
        print("Running comment3 robustness evaluation:")
        print(" ".join(eval_cmd))
        try:
            subprocess.run(eval_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Warning] comment3 robustness evaluation failed (exit={e.returncode}).")
        except FileNotFoundError:
            print("[Warning] comment3 script not found. Check --comment3_script path.")


if __name__ == "__main__":
    main()
