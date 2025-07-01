import torch
import torch.nn as nn
from xresnet1d_101 import xresnet1d101
class KEDEncoderWrapper(nn.Module):
    def __init__(self, ked_encoder, token_num=100):
        super().__init__()
        self.ked_encoder = ked_encoder
        self.adapt_pool = nn.AdaptiveAvgPool1d(token_num)
       

        # 记录模型关键配置参数
        self.hidden_size = 768   # 按你的模型实际输出设定
        self.token_num = token_num


        # 参数全部冻结（不参与训练）
        for p in self.ked_encoder.parameters():
            p.requires_grad = False
    

    @torch.no_grad()  # 前向推理不计算梯度
    def forward(self, x):
        feats = self.ked_encoder(x)           # [batch, 768, 157]
        token_emb = self.adapt_pool(feats)    # [batch, 768, 100]
        token_emb = token_emb.transpose(1,2)  # [batch, 100, 768]
        global_emb = token_emb.mean(dim=1)    # [batch, 768]
        return global_emb, token_emb

if __name__ == '__main__':
    ked_encoder = xresnet1d101(input_channels=12, kernel_size=5, use_ecgNet_Diagnosis='other')
   
    # 包装成兼容ECG-Chat格式
    ecg_encoder = KEDEncoderWrapper(ked_encoder)
    # forward推理
    x = torch.randn(8, 12, 5000)  # 假设batch=8, 12导联, 5000采样点
    global_emb, token_emb = ecg_encoder(x)
    #global_emb=ecg_encoder(x)
    print(global_emb.shape)  # torch.Size([8, 768])
    print(token_emb.shape)   # torch.Size([8, 100, 768])

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from xresnet1d_101 import xresnet1d101
# import torch.nn.functional as F

# def get_soft_label(logits, T=1.0):
#         """
#         logits: [batch, num_classes]
#         T: temperature（float，T>0，通常取2~8）
#         返回：soft_label（概率分布）
#         """
#         soft_label = F.softmax(logits / T, dim=-1)
#         return soft_label

# class KEDEncoderWithSoftLabel(nn.Module):
#     def __init__(self, num_classes=105, token_num=100, input_channels=12, kernel_size=5):
#         super().__init__()
#         # 只实例化一次xresnet1d101（必须带分类头，即use_ecgNet_Diagnosis='ecgNet'）
#         self.backbone = xresnet1d101(
#             input_channels=input_channels,
#             kernel_size=kernel_size,
#             num_classes=num_classes,
#             use_ecgNet_Diagnosis='ecgNet'
#         )
#         self.token_num = token_num
#         self.adapt_pool = nn.AdaptiveAvgPool1d(token_num)
#         # 自动获取hidden_size
#         # 假设最后block输出特征数等于分类头输入
#         self.hidden_size = self.backbone[-2][-1].out_channels if hasattr(self.backbone[-2][-1], 'out_channels') else 768



#     def forward(self, x):
#         # 1. 前向到分类头前，获得主干特征
#         # 主干部分: 到最后一层head之前
#         # 具体层数视你的xresnet1d101定义而定，下面假定head是self.backbone[-1]
#         feats = x
#         for layer in list(self.backbone.children())[:-1]:
#             feats = layer(feats)  # [B, C, L]
#         # 2. 分类头输出soft label
#         logits = self.backbone[-1](feats)           # [B, num_classes]
#         soft_label = get_soft_label(logits, T=4.0)      # [B, num_classes]
#         # 3. KED风格池化和特征
#         token_emb = self.adapt_pool(feats)          # [B, C, token_num]
#         token_emb = token_emb.transpose(1, 2)       # [B, token_num, C]
#         global_emb = token_emb.mean(dim=1)          # [B, C]
#         return soft_label, global_emb, token_emb

# if __name__ == "__main__":
#     model = KEDEncoderWithSoftLabel(num_classes=105, token_num=100)
#     x = torch.randn(4, 12, 5000)
#     soft_label, global_emb, token_emb = model(x)
#     print('soft_label:', soft_label.shape)     # [4, 105]
#     print('global_emb:', global_emb.shape)     # [4, 768]（或其它hidden_size）
#     print('token_emb:', token_emb.shape)       # [4, 100, 768]