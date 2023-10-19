import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer, num_kp, num_bins):
        super( MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

        self.fc_roll = nn.Linear(descriptor_nc, num_bins)
        self.fc_pitch = nn.Linear(descriptor_nc, num_bins)
        self.fc_yaw = nn.Linear(descriptor_nc, num_bins)
        self.fc_t = nn.Linear(descriptor_nc, 3)
        self.fc_exp = nn.Linear(descriptor_nc, 3*num_kp)

    def forward(self, input_3dmm):
        out = self.first(input_3dmm) # input_3dmm:torch.Size([1, 73, 27]),out:torch.Size([1, 1024, 21])
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:,:,3:-3] # torch.Size([1, 1024, 15]), torch.Size([1, 1024, 9]), torch.Size([1, 1024, 3])
        out = self.pooling(out) # torch.Size([1, 1024, 1])
        out = out.view(out.shape[0], -1) # torch.Size([1, 1024])
        #print('out:', out.shape)

        yaw = self.fc_yaw(out) # torch.Size([1, 66])
        pitch = self.fc_pitch(out) # torch.Size([1, 66])
        roll = self.fc_roll(out) # torch.Size([1, 66])
        t = self.fc_t(out) # torch.Size([1, 3])
        exp = self.fc_exp(out) # torch.Size([1, 45]) 

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp} 