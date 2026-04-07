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
#  - Module Name : train_search
#  - Function    : DARTS architecture search training loop
#
#

"""DARTS architecture search loop for the drone IQ classification task."""

import argparse
import csv
import pickle

import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split

from architect import Architect
from fileio import WaveformDataset
from model_search import Network
from utils import AverageMeter, EarlyStopping, accuracy


def main():
    parser = argparse.ArgumentParser("DRONE-IQ DARTS Search")
    parser.add_argument('--train_dir', type=str, default='/home/dykim/drone/dronedetect/train',
                        help='Training data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025,
                        help='SGD learning rate for the network weights')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='Weight decay applied to the network weights')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4,
                        help='Learning rate for the architecture parameters (alphas)')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3,
                        help='Weight decay for the architecture parameters')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of architecture search epochs')
    parser.add_argument('--init_channels', type=int, default=4,
                        help='Initial channel count of the network')
    parser.add_argument('--layers', type=int, default=2,
                        help='Total number of cells (layers) used during search')
    parser.add_argument('--classes', type=int, default=13, help='Number of output classes')
    parser.add_argument('--train_portion', type=float, default=0.8,
                        help='Fraction of the train split used for weight updates '
                             '(the rest is used as the architecture validation set)')
    parser.add_argument('--unrolled', action='store_true',
                        help='(Advanced) enable second-order unrolled bilevel optimization')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--threads', type=int, default=4, help='Number of CPU threads')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (in epochs)')
    parser.add_argument('--genotype_out', type=str, default=None,
                        help='Path to save the best genotype as a pickle file')
    parser.add_argument('--log_out', type=str, default='alpha_loss_log.csv',
                        help='Path to save the alpha/loss CSV log')

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cuda")

    # ===== Data =====
    dataset = WaveformDataset(args.train_dir)
    train_size = int(len(dataset) * args.train_portion)
    valid_size = len(dataset) - train_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size,
                                               shuffle=False, num_workers=8)

    # ===== Model / optimizers =====
    model = Network(args.init_channels, args.classes, args.layers)
    model = model.to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {num_parameters}")

    criterion = torch.nn.CrossEntropyLoss()
    architect = Architect(model, criterion, args.arch_learning_rate, args.arch_weight_decay)
    optimizer = torch.optim.SGD(model.weights(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    # Early stopping captures both the best weights and the best genotype
    early_stopping = EarlyStopping(patience=args.patience, save_genotype=True)

    # CSV header
    csv_path = args.log_out
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'alpha_mean', 'train_loss', 'val_loss'])

    # ===== Training loop =====
    for epoch in range(args.epochs):
        model.train()
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        valid_loss_meter = AverageMeter()
        valid_acc_meter = AverageMeter()
        valid_iter = iter(valid_loader)

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.weights(), args.grad_clip)
            optimizer.step()
            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            train_loss_meter.update(loss.item(), targets.size(0))
            train_acc_meter.update(prec1, targets.size(0))

            try:
                input_val, target_val = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_loader)
                input_val, target_val = next(valid_iter)
            input_val = input_val.to(device)
            target_val = target_val.to(device)
            architect.step(inputs, targets, input_val, target_val, args.unrolled)

        # ----- Validation -----
        model.eval()
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                prec1 = accuracy(outputs, targets, topk=(1,))[0]
                valid_loss_meter.update(val_loss.item(), targets.size(0))
                valid_acc_meter.update(prec1, targets.size(0))

        # Mean of the normal-cell alphas, useful for monitoring search dynamics
        alpha_mean = model.arch_parameters()[0].mean().item()

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, alpha_mean,
                             train_loss_meter.avg, valid_loss_meter.avg])

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss_meter.avg:.3f}, Train Acc: {train_acc_meter.avg:.2f}% | "
              f"Val Loss: {valid_loss_meter.avg:.3f}, Val Acc: {valid_acc_meter.avg:.2f}% | "
              f"Alpha Mean: {alpha_mean:.4f}")

        early_stopping(valid_loss_meter.avg, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model_state)
            final_genotype = early_stopping.best_genotype
            print(f"Best Validation Loss: {early_stopping.best_val_loss:.3f}")
            print(f"Final searched genotype: {final_genotype}")
            print("Copy this genotype into the NASGenotype variable in genotypes.py.")
            break

        genotype = model.genotype()
        print(f"Genotype: {genotype}")

    # When the loop completes without early stopping, report the last genotype
    if not early_stopping.early_stop:
        final_genotype = model.genotype()
        print(f"Final searched genotype: {final_genotype}")
        print("Copy this genotype into the NASGenotype variable in genotypes.py.")

    # Optional: persist the best genotype as a pickle for downstream tools
    if args.genotype_out and early_stopping.best_genotype is not None:
        with open(args.genotype_out, 'wb') as f:
            pickle.dump(early_stopping.best_genotype, f)

    # ===== Post-training plots =====
    epochs, alphas, train_losses, val_losses = [], [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            alphas.append(float(row['alpha_mean']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))

    plt.figure()
    plt.plot(epochs, alphas, label='Alpha Mean')
    plt.xlabel('Epoch')
    plt.ylabel('Alpha Mean')
    plt.title('Alpha Mean over Epochs')
    plt.legend()
    plt.savefig('alpha_mean_plot.png')

    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')


if __name__ == "__main__":
    main()
