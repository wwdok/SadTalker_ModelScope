import os

from tqdm import tqdm
import torch
import numpy as np
import random
import scipy.io as scio
import src.utils.audio as audio

def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

def parse_audio_length(audio_length, sr, fps):
    # 根据视频FPS等参数计算需要这段音频对应几帧画面
    bit_per_frames = sr / fps # 每秒采样点数/每秒播放几帧 = 每帧采样多少点数

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames

def generate_blink_seq(num_frames):
    ratio = np.zeros((num_frames,1))
    frame_id = 0
    while frame_id in range(num_frames):
        start = 80
        if frame_id+start+9<=num_frames - 1:
            ratio[frame_id+start:frame_id+start+9, 0] = [0.5,0.6,0.7,0.9,1, 0.9, 0.7,0.6,0.5]
            frame_id = frame_id+start+9
        else:
            break
    return ratio 

def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames,1))
    if num_frames<=20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10,num_frames), min(int(num_frames/2), 70))) 
        if frame_id+start+5<=num_frames - 1:
            ratio[frame_id+start:frame_id+start+5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            frame_id = frame_id+start+5
        else:
            break
    return ratio

def get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=False, idlemode=False, length_of_audio=False, use_blink=True):

    syncnet_mel_step_size = 16
    fps = 25

    pic_name = os.path.splitext(os.path.split(first_coeff_path)[-1])[0]
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]

    
    if idlemode:
        num_frames = int(length_of_audio * 25)
        indiv_mels = np.zeros((num_frames, 80, 16))
    else:
        wav = audio.load_wav(audio_path, 16000) 
        wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
        print(f"==>> len(wav): {len(wav)}") # 87552
        print(f"==>> wav_length: {wav_length}") # : 87040
        print(f"==>> num_frames: {num_frames}") # 136
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = audio.melspectrogram(wav).T
        print(f"==>> orig_mel.shape: {orig_mel.shape}") # (436, 80)，436 ≈ 87040/sfft hop_size = 87040/200
        spec = orig_mel.copy()
        indiv_mels = []

        for i in tqdm(range(num_frames), 'mel:'):
            start_frame_num = i-2
            start_idx = int(80. * (start_frame_num / float(fps)))
            # print(f"==>> start_idx: {start_idx}") # 从-6开始，步长为+3
            end_idx = start_idx + syncnet_mel_step_size
            # print(f"==>> end_idx: {end_idx}") # 从10开始，步长为+3
            seq = list(range(start_idx, end_idx)) # 长度为syncnet_mel_step_size=16，这也是张量形状里的16
            seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
            m = spec[seq, :]
            # print(f"==>> m.shape: {m.shape}") # (16, 80)
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels)         # T 80 16
        print(f"==>> indiv_mels.shape: {indiv_mels.shape}") # (136, 80, 16) # 这个张量意味着每一帧图片由(80,16)的音频特征决定

    ratio = generate_blink_seq_randomly(num_frames)      # T
    print(f"==>> ratio.shape: {ratio.shape}") #  (136, 1)
    # print(f"==>> ratio: {ratio}") # 里面大多数值是0.，有两组 [0.5][0.9][1. ][0.9][0.5]
    source_semantics_path = first_coeff_path
    source_semantics_dict = scio.loadmat(source_semantics_path)
    ref_coeff = source_semantics_dict['coeff_3dmm'][:1,:70]         #1 70
    # 将第一帧的参考系数赋值T遍
    ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)

    if ref_eyeblink_coeff_path is not None:
        ratio[:num_frames] = 0
        refeyeblink_coeff_dict = scio.loadmat(ref_eyeblink_coeff_path)
        refeyeblink_coeff = refeyeblink_coeff_dict['coeff_3dmm'][:,:64]
        refeyeblink_num_frames = refeyeblink_coeff.shape[0]
        if refeyeblink_num_frames<num_frames:
            div = num_frames//refeyeblink_num_frames
            re = num_frames%refeyeblink_num_frames
            refeyeblink_coeff_list = [refeyeblink_coeff for i in range(div)]
            refeyeblink_coeff_list.append(refeyeblink_coeff[:re, :64])
            refeyeblink_coeff = np.concatenate(refeyeblink_coeff_list, axis=0)
            print(f"==>> refeyeblink_coeff.shape[0]: {refeyeblink_coeff.shape[0]}")

        ref_coeff[:, :64] = refeyeblink_coeff[:num_frames, :64] 
    
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0) # bs T 1 80 16

    if use_blink:
        ratio = torch.FloatTensor(ratio).unsqueeze(0)                       # bs T
    else:
        ratio = torch.FloatTensor(ratio).unsqueeze(0).fill_(0.) 
                               # bs T
    ref_coeff = torch.FloatTensor(ref_coeff).unsqueeze(0)                # bs T 70

    indiv_mels = indiv_mels.to(device)
    print(f"==>> indiv_mels.shape: {indiv_mels.shape}") # torch.Size([1, 136, 1, 80, 16])
    ratio = ratio.to(device)
    print(f"==>> ratio.shape: {ratio.shape}") # torch.Size([1, 136, 1])
    ref_coeff = ref_coeff.to(device)
    print(f"==>> ref_coeff.shape: {ref_coeff.shape}") # torch.Size([1, 136, 70])

    return {'indiv_mels': indiv_mels,  
            'ref': ref_coeff, 
            'num_frames': num_frames, 
            'ratio_gt': ratio,
            'audio_name': audio_name, 'pic_name': pic_name}

