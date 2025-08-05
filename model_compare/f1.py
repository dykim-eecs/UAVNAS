import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from fileio import WaveformDataset
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ===== Dataset 설정 =====
test_dir = '/home/isp/drone_project/data/test'
test_dataset = WaveformDataset(test_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# ===== 모델 리스트 =====
models = {
    "UAVNAS": "model.onnx",
    "ResNet50": "resnet50.onnx",
    "MobileNet": "mobilenet.onnx",
    "ShuffleNet": "shufflenet.onnx",
    "2DCNN": "2dcnn.onnx"
}

# ===== 클래스 라벨 =====
classes = [f"D{i+1}" for i in range(13)]

# ===== F1 스코어 저장 리스트 =====
results = []

# ===== 각 모델별 Confusion Matrix & F1 Score =====
for model_name, model_path in models.items():
    print(f"\n===== Processing Model: {model_name} =====")

    # ONNX 모델 로드
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 예측 및 라벨 저장
    y_true = []
    y_pred = []

    for inputs, targets in test_loader:
        inputs_np = inputs.numpy()
        outputs = session.run(None, {input_name: inputs_np})[0]
        pred = np.argmax(outputs, axis=1)
        y_true.extend(targets.numpy())
        y_pred.extend(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ===== Confusion Matrix =====
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='.3f', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        cbar=True, cbar_kws={'label': 'Normalized Probability (0-1)'},
        annot_kws={"size": 10, "weight": "bold"}
    )
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold', labelpad=15)
    plt.ylabel('Actual Class', fontsize=14, fontweight='bold', labelpad=15)
    plt.title(f'Normalized Confusion Matrix for {model_name}', fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()

    cm_filename = f'confusion_matrix_{model_name.lower()}.png'
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion Matrix saved as '{cm_filename}'")

    # ===== F1 Score =====
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    results.append({
        "Model": model_name,
        "F1_Macro": f1_macro,
        "F1_Micro": f1_micro,
        "F1_Weighted": f1_weighted
    })

# ===== CSV 저장 =====
df = pd.DataFrame(results)
df.to_csv("f1_scores.csv", index=False)
print("\nF1 scores saved to 'f1_scores.csv'")

# ===== 세로형 막대 그래프 =====
plt.figure(figsize=(8, 6))
bar_width = 0.25
x = np.arange(len(df["Model"]))

plt.bar(x - bar_width, df["F1_Macro"], width=bar_width, label="Macro")
plt.bar(x, df["F1_Micro"], width=bar_width, label="Micro")
plt.bar(x + bar_width, df["F1_Weighted"], width=bar_width, label="Weighted")

plt.xticks(x, df["Model"])
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.title("F1 Score Comparison (Macro, Micro, Weighted)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("f1_scores_vertical.png", dpi=300, bbox_inches='tight')
plt.show()

print("F1 score comparison plot saved as 'f1_scores_vertical.png'")
