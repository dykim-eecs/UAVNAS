import os
import numpy as np
import torch
from torch.utils.data import Dataset

class WaveformDataset(Dataset):
    """
    드론 IQ 데이터를 불러오는 Dataset 클래스 (.fc32, float32).
    파일 하나당 IQ쌍을 사용하여 (2, 32, 3072) 형태로 리턴.
    정규화를 적용하여 학습 안정성 향상.
    """
    def __init__(self, root_dir):
        self.files = []
        self.labels = []

        for class_name in sorted(os.listdir(root_dir)):  # '00', '01', ..., '12'
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

        assert len(self.files) == len(self.labels), "파일과 레이블 수 mismatch"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        # .fc32 파일을 float32 형식으로 로드
        data = np.fromfile(file_path, dtype=np.float32)

        try:
            iq_pairs = data.reshape(-1, 2)
            
        except:
            raise ValueError(f"파일 {file_path}의 크기가 예상과 다릅니다.")

        iq_pairs = iq_pairs[:98304, :]
        iq_tensor = torch.from_numpy(iq_pairs).T.view(2, 32, 3072)
        # 정규화
        iq_tensor = (iq_tensor - iq_tensor.mean()) / (iq_tensor.std() + 1e-6)
        return iq_tensor, label