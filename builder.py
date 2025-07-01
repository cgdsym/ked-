import os

import sys
# 这里换成 ked_encoder 实际所在的上级目录路径 
sys.path.append("/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/model/multimodal_encoder")
# 新增：导入你自定义的编码器
from ked_encoder import KEDEncoderWrapper,KEDEncoderWithSoftLabel
from xresnet1d_101 import xresnet1d101
import torch



def build_ecg_tower(pretrained_path=None, device='cuda'):
    ecg_encoder = KEDEncoderWithSoftLabel(num_classes=105, token_num=100)
    if pretrained_path:
        ecg_encoder.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    ecg_encoder = ecg_encoder.to(device)
    return ecg_encoder