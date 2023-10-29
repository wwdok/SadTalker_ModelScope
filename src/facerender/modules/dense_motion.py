from torch import nn
import torch.nn.functional as F
import torch
from src.facerender.modules.util import Hourglass, make_coordinate_grid, kp2gaussian

from src.facerender.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False):
        super(DenseMotionNetwork, self).__init__()
        # self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(feature_channel+1), max_features=max_features, num_blocks=num_blocks)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = BatchNorm3d(compress, affine=True)

        if estimate_occlusion_map:
            # self.occlusion = nn.Conv2d(reshape_channel*reshape_depth, 1, kernel_size=7, padding=3)
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        self.num_kp = num_kp


    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape # torch.Size([1, 4, 16, 64, 64])
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['value'].type()) # torch.Size([16, 64, 64, 3])
        identity_grid = identity_grid.view(1, 1, d, h, w, 3) # 按理说grid只有5个维度，这里有6个维度，前面多出来的两个维度从下面看是batch size和num_kp的含义
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3) # torch.Size([1, 15, 16, 64, 64, 3])
        
        # if 'jacobian' in kp_driving:
        if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)                  


        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 1, 3) # torch.Size([1, 15, 16, 64, 64, 3]), (bs, num_kp, d, h, w, 3)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1) # background作为一个伪kp
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1) # torch.Size([1, 16, 16, 64, 64, 3]), bs num_kp+1 d h w 3
        
        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions): # feature: torch.Size([1, 4, 16, 64, 64]), sparse_motions: torch.Size([1, 16, 16, 64, 64, 3])
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w), torch.Size([1, 16, 1, 4, 16, 64, 64])，相当于每个3d kp都分配给它一个3d head feature
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w),   torch.Size([16, 4,  16, 64, 64])
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3),   torch.Size([16, 16, 64, 64, 3])
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions) # torch.Size([16, 4, 16, 64, 64])
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w),    torch.Size([1, 16, 4, 16, 64, 64])
        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source): # torch.Size([1, 16, 4, 16, 64, 64]), torch.Size([1, 15, 3]),torch.Size([1, 15, 3])
        spatial_size = feature.shape[3:] # torch.Size([16, 64, 64])，该函数内部实际没有对feature做处理
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01) # torch.Size([1, 15, 16, 64, 64])
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)   # torch.Size([1, 15, 16, 64, 64])
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.type()) # torch.Size([1, 1, 16, 64, 64])
        heatmap = torch.cat([zeros, heatmap], dim=1) # torch.Size([1, 16, 16, 64, 64])
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)，扩增出C维度
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape # torch.Size([1, 32, 16, 64, 64])，可以想象成有16层特征图，每层从鼻子切到后脑勺，每层特征图的形状是32*64*64

        feature = self.compress(feature) # torch.Size([1, 4, 16, 64, 64])，通道数压缩到4了，不过还有16层特征图
        feature = self.norm(feature)
        feature = F.relu(feature) # torch.Size([1, 4, 16, 64, 64])

        out_dict = dict()
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source) # torch.Size([1, 16, 16, 64, 64, 3])，生成了一个后面会用作grid的张量，不过grid最多只有5维，这里有6维
        deformed_feature = self.create_deformed_feature(feature, sparse_motion) # torch.Size([1, 16, 4, 16, 64, 64])，里面核心就是做了grid sample操作

        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source) # torch.Size([1, 16, 1, 16, 64, 64])

        input_ = torch.cat([heatmap, deformed_feature], dim=2) # torch.Size([1, 16, 5, 16, 64, 64])，在（N，KP，C，D，H，W）的C维度拼接
        input_ = input_.view(bs, -1, d, h, w) # torch.Size([1, 80, 16, 64, 64])，KP和C维度融合到一个维度

        # input = deformed_feature.view(bs, -1, d, h, w)      # (bs, num_kp+1 * c, d, h, w)

        prediction = self.hourglass(input_) # torch.Size([1, 112, 16, 64, 64])


        mask = self.mask(prediction) # torch.Size([1, 16, 16, 64, 64])
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)  # torch.Size([1, 16, 1, 16, 64, 64]), (bs, num_kp+1, 1, d, h, w)
        
        zeros_mask = torch.zeros_like(mask)   
        mask = torch.where(mask < 1e-3, zeros_mask, mask) 

        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w),torch.Size([1, 16, 3, 16, 64, 64])
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w) # torch.Size([1, 1792, 64, 64])
            occlusion_map = torch.sigmoid(self.occlusion(prediction)) # torch.Size([1, 1, 64, 64])
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
