import torch
from torch import nn
from src.audio2pose_models.cvae import CVAE
from src.audio2pose_models.discriminator import PoseSequenceDiscriminator
from src.audio2pose_models.audio_encoder import AudioEncoder

class Audio2Pose(nn.Module):
    def __init__(self, cfg, wav2lip_checkpoint, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.MODEL.CVAE.SEQ_LEN # 32
        self.latent_dim = cfg.MODEL.CVAE.LATENT_SIZE # 64
        self.device = device

        self.audio_encoder = AudioEncoder(wav2lip_checkpoint, device)
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.netG = CVAE(cfg)
        self.netD_motion = PoseSequenceDiscriminator(cfg) # 该代码仓库暂时没用到，所以该代码仓库暂时没有训练代码
        
        
    def forward(self, x):

        batch = {}
        coeff_gt = x['gt'].cuda().squeeze(0)           #bs frame_len+1 73
        batch['pose_motion_gt'] = coeff_gt[:, 1:, 64:70] - coeff_gt[:, :1, 64:70] #bs frame_len 6
        batch['ref'] = coeff_gt[:, 0, 64:70]  #bs  6
        batch['class'] = x['class'].squeeze(0).cuda() # bs
        indiv_mels= x['indiv_mels'].cuda().squeeze(0) # bs seq_len+1 80 16

        # forward
        audio_emb_list = []
        audio_emb = self.audio_encoder(indiv_mels[:, 1:, :, :].unsqueeze(2)) #bs seq_len 512
        batch['audio_emb'] = audio_emb
        batch = self.netG(batch)

        pose_motion_pred = batch['pose_motion_pred']           # bs frame_len 6
        pose_gt = coeff_gt[:, 1:, 64:70].clone()               # bs frame_len 6
        pose_pred = coeff_gt[:, :1, 64:70] + pose_motion_pred  # bs frame_len 6

        batch['pose_pred'] = pose_pred
        batch['pose_gt'] = pose_gt

        return batch

    def test(self, x):

        batch = {}
        ref = x['ref'] #  torch.Size([1, 136, 70])                          #bs 1 70
        batch['ref'] = x['ref'][:,0,-6:]  # torch.Size([1, 6]), 去掉64个表情系数，只用到后6个姿态系数
        batch['class'] = x['class']  
        bs = ref.shape[0]
        
        indiv_mels= x['indiv_mels']               # bs T 1 80 16, torch.Size([1, 136, 1, 80, 16])
        indiv_mels_use = indiv_mels[:, 1:]        # we regard the ref as the first frame, torch.Size([1, 135, 1, 80, 16])
        num_frames = x['num_frames']
        num_frames = int(num_frames) - 1 # 135

        #  
        div = num_frames//self.seq_len # 4
        re = num_frames%self.seq_len # 7
        audio_emb_list = []
        pose_motion_pred_list = [torch.zeros(batch['ref'].unsqueeze(1).shape, dtype=batch['ref'].dtype, 
                                                device=batch['ref'].device)]
        # 先处理整除部分
        for i in range(div):
            z = torch.randn(bs, self.latent_dim).to(ref.device) # torch.Size([1, 64])
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:]) #torch.Size([1, 32, 512])，bs seq_len 512
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch['pose_motion_pred'])  #list of bs seq_len 6
        # 再处理余数部分
        if re != 0:
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, -1*self.seq_len:,:,:,:]) #bs seq_len  512
            if audio_emb.shape[1] != self.seq_len:
                pad_dim = self.seq_len-audio_emb.shape[1]
                pad_audio_emb = audio_emb[:, :1].repeat(1, pad_dim, 1) 
                audio_emb = torch.cat([pad_audio_emb, audio_emb], 1) 
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch['pose_motion_pred'][:,-1*re:,:])   
        
        pose_motion_pred = torch.cat(pose_motion_pred_list, dim = 1) # torch.Size([1, 136, 6])
        batch['pose_motion_pred'] = pose_motion_pred

        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # bs T 6，可见预测的是相对初始姿态的变化量

        batch['pose_pred'] = pose_pred
        return batch
