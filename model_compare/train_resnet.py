# train_resnet.py (pretrained=False -> weights=None, add ONNX export at the end)
import os
import torch
import argparse
from torch.utils.data import random_split
from fileio import WaveformDataset
from utils import AverageMeter, accuracy
import torchvision.models as models

def main():
    parser = argparse.ArgumentParser("ResNet50 Training")
    parser.add_argument('--train_dir', type=str, default='data/train', help='학습용 데이터 디렉토리')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='weights SGD 학습률')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD 모멘텀')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weights에 대한 가중치 감쇠')
    parser.add_argument('--epochs', type=int, default=50, help='아키텍처 탐색 epoch 수')
    parser.add_argument('--classes', type=int, default=13, help='클래스 개수')
    parser.add_argument('--train_portion', type=float, default=0.8, help='train 데이터 중 가중치 학습에 사용할 비율')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping 값')
    parser.add_argument('--threads', type=int, default=4, help='CPU 사용 스레드 수')
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = WaveformDataset(args.train_dir)
    train_size = int(len(dataset) * args.train_portion)
    valid_size = len(dataset) - train_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.resnet50(weights=None, num_classes=args.classes)
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    avg_acc_list = []
    for epoch in range(args.epochs):
        model.train()
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        valid_loss_meter = AverageMeter()
        valid_acc_meter = AverageMeter()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            train_loss_meter.update(loss.item(), targets.size(0))
            train_acc_meter.update(prec1, targets.size(0))

        model.eval()
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                prec1 = accuracy(outputs, targets, topk=(1,))[0]
                valid_loss_meter.update(val_loss.item(), targets.size(0))
                valid_acc_meter.update(prec1, targets.size(0))

        avg_acc_list.append(valid_acc_meter.avg)
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss_meter.avg:.3f}, Train Acc: {train_acc_meter.avg:.2f}% | "
              f"Val Loss: {valid_loss_meter.avg:.3f}, Val Acc: {valid_acc_meter.avg:.2f}%")

    torch.save(model.state_dict(), 'resnet50.pt')
    overall_avg_acc = sum(avg_acc_list) / len(avg_acc_list) if avg_acc_list else 0
    print(f"Overall Avg Acc: {overall_avg_acc:.2f}%")

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 2, 32, 3072).to(device)
    torch.onnx.export(model, dummy_input, 'resnet50.onnx', opset_version=11)
    print("Exported to resnet50.onnx")

    return overall_avg_acc

if __name__ == "__main__":
    main()