import torch
import torch.nn.functional as F
from torch import nn

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, use_act = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual
        self.use_act = use_act

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        
        if self.use_act:
            return self.act(out)
        else:
            return out

class SimpleWrapperV2(nn.Module):
    """
    用于exp to expression，对应论文中Fig 3中的Audio Encoder和Mapping
    """
    def __init__(self) -> None:
        super().__init__()
        # 这个audio encoder跟PoseVAE中的AudioEncoder模型结构一米一样，都是把每一时间步、形状为$(80, 16)$的音频梅尔谱图编码成$(512)$的向量，但可以看出，这两个模型不共享权重。
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )

        #### load the pre-trained audio_encoder 
        #self.audio_encoder = self.audio_encoder.to(device)  
        '''
        wav2lip_state_dict = torch.load('/apdcephfs_cq2/share_1290939/wenxuazhang/checkpoints/wav2lip.pth')['state_dict']
        state_dict = self.audio_encoder.state_dict()

        for k,v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                print('init:', k)
                state_dict[k.replace('module.audio_encoder.', '')] = v
        self.audio_encoder.load_state_dict(state_dict)
        '''

        self.mapping1 = nn.Linear(512+64+1, 64)
        #self.mapping2 = nn.Linear(30, 64)
        #nn.init.constant_(self.mapping1.weight, 0.)
        nn.init.constant_(self.mapping1.bias, 0.)

    def forward(self, x, ref, ratio):
        x = self.audio_encoder(x).view(x.size(0), -1) # audio_encoder把x从形状[BS*T 1 80 16]编码成形状[BS*T,512]
        # 这里T=10，它其实来自于for 循环的步长，含义跟batch size 差不多
        # print(f"==>> x.shape: {x.shape}") # torch.Size([10, 512])
        ref_reshape = ref.reshape(x.size(0), -1)
        # print(f"==>> ref_reshape.shape: {ref_reshape.shape}") # torch.Size([10, 64])
        ratio = ratio.reshape(x.size(0), -1)
        # print(f"==>> ratio.shape: {ratio.shape}") # torch.Size([10, 1])
        
        y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1)) 
        # print(f"==>> y.shape: {y.shape}") # torch.Size([10, 64])
        out = y.reshape(ref.shape[0], ref.shape[1], -1) #+ ref # resudial
        # print(f"==>> out.shape: {out.shape}") # torch.Size([1, 10, 64])
        return out
