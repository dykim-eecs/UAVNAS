# measure.py (modified to use Model_size instead of Storage, save to CSV, and handle all models)
import torch
import time
import os
from ptflops import get_model_complexity_info
import torchvision.models as models
import pandas as pd
from collections import namedtuple

# From genotypes.py
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
NASGenotype = Genotype(normal=[('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('max_pool_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2), ('sep_conv_5x5', 3), ('dil_conv_5x5', 4), ('skip_connect', 1)], reduce_concat=[2, 3, 4, 5])

# From operations.py (simplified for measure, assuming OPS and helpers defined)
class ReLUConvBN(torch.nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            torch.nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x):
        return self.op(x)

class SepConv(torch.nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            torch.nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(C_in, affine=affine),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(C_in, C_in, kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            torch.nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x):
        return self.op(x)

class DilConv(torch.nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            torch.nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x):
        return self.op(x)

class Zero(torch.nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
    def forward(self, x):
        if self.stride == 1:
            return torch.zeros_like(x)
        else:
            return torch.zeros(
                (x.size(0), x.size(1), x.size(2) // self.stride, x.size(3) // self.stride),
                dtype=x.dtype, device=x.device)

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class FactorizedReduce(torch.nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0, "C_out must be divisible by 2"
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv1 = torch.nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out1 = self.conv1(x)
        out2 = self.conv2(x)  # 안전하게 변경 (오프셋 제거)
        out = torch.cat([out1, out2], dim=1)
        out = self.bn(out)
        return out

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: torch.nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: torch.nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine)
}

class Cell(torch.nn.Module):
    def __init__(self, genotype, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)

        self._steps = steps
        self._multiplier = multiplier
        self.ops_by_node = torch.nn.ModuleList()
        self.indices_by_node = []
        for i in range(steps):
            op_name1, inp1 = genotype[2 * i]
            op_name2, inp2 = genotype[2 * i + 1]
            stride1 = 2 if reduction and inp1 < 2 else 1
            stride2 = 2 if reduction and inp2 < 2 else 1
            op1 = OPS[op_name1](C, stride1, affine=True)
            op2 = OPS[op_name2](C, stride2, affine=True)
            self.ops_by_node.append(torch.nn.ModuleList([op1, op2]))
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

class NetworkFixed(torch.nn.Module):
    def __init__(self, genotype, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3):
        super(NetworkFixed, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        input_channels = 2  # float32 IQ 데이터이므로 채널 2개
        C_cur = C * stem_multiplier
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, C_cur, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(C_cur)
        )

        self.cells = torch.nn.ModuleList()
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
            cell = Cell(cell_genotype, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            C_prev_prev = C_prev
            C_prev = C_curr * multiplier
            if reduction:
                C = C * 2

        self.global_pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = s0
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

class Simple2DCNN(torch.nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(128 * 4 * 384, num_classes)
    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def measure(model, pt_path, onnx_path, avg_acc):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pt_path and os.path.exists(pt_path):
        model.load_state_dict(torch.load(pt_path))
    model = model.to(device)
    model.eval()
    flops, params = get_model_complexity_info(model, (2, 32, 3072), as_strings=False)
    macs = flops / 2
    model_size = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0 # in bytes
    dummy_input = torch.randn(1, 2, 32, 3072).to(device)
    times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        times.append((time.time() - start) * 1000)
    inference_time = sum(times) / len(times)
    return {
        'MACs': macs,
        'Params': params,
        'Model_size (bytes)': model_size,
        'Avg_acc (%)': avg_acc,
        'Inference_Time (ms)': inference_time
    }

# Initialize models
resnet = models.resnet50(weights=None, num_classes=13)
resnet.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
mobilenet = models.mobilenet_v3_small(weights=None, num_classes=13)
mobilenet.features[0][0] = torch.nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=False)
shufflenet = models.shufflenet_v2_x1_0(weights=None, num_classes=13)
shufflenet.conv1[0] = torch.nn.Conv2d(2, 24, kernel_size=3, stride=2, padding=1, bias=False)
twodcnn = Simple2DCNN(13)
uavnas = NetworkFixed(NASGenotype, C=4, num_classes=13, layers=2)

# Manually input Avg_acc from training outputs
avg_accs = {
    'ResNet50': 0.0, # Replace with actual
    'MobileNet': 0.0,
    'ShuffleNet': 0.0,
    '2DCNN': 0.0,
    'UAVNAS': 0.0
}

results = {}
results['ResNet50'] = measure(resnet, 'resnet50.pt', 'resnet50.onnx', avg_accs['ResNet50'])
results['MobileNet'] = measure(mobilenet, 'mobilenet.pt', 'mobilenet.onnx', avg_accs['MobileNet'])
results['ShuffleNet'] = measure(shufflenet, 'shufflenet.pt', 'shufflenet.onnx', avg_accs['ShuffleNet'])
results['2DCNN'] = measure(twodcnn, '2dcnn.pt', '2dcnn.onnx', avg_accs['2DCNN'])
results['UAVNAS'] = measure(uavnas, None, 'model.onnx', avg_accs['UAVNAS'])

df = pd.DataFrame(results).T
df.to_csv('model_results.csv', index_label='Model')
print("Results saved to model_results.csv")
print(df)