import os
import numpy as np
from PIL import Image
from skimage import io, img_as_float32, transform
import torch
import scipy.io as scio

def get_facerender_data(coeff_path, pic_path, first_coeff_path, audio_path, 
                        batch_size, input_yaw_list=None, input_pitch_list=None, input_roll_list=None, 
                        expression_scale=1.0, still_mode = False, preprocess='crop', size = 256):

    semantic_radius = 13
    video_name = os.path.splitext(os.path.split(coeff_path)[-1])[0]
    txt_path = os.path.splitext(coeff_path)[0]

    data={}

    img1 = Image.open(pic_path) # './results/2023_10_17_20_02_35/first_frame_dir/man.png'
    source_image = np.array(img1)
    source_image = img_as_float32(source_image)
    source_image = transform.resize(source_image, (size, size, 3))
    source_image = source_image.transpose((2, 0, 1)) # shape:(3, 256, 256)
    source_image_ts = torch.FloatTensor(source_image).unsqueeze(0)
    source_image_ts = source_image_ts.repeat(batch_size, 1, 1, 1)
    data['source_image'] = source_image_ts
 
    source_semantics_dict = scio.loadmat(first_coeff_path) # coeff_3dmm:(1, 73);full_3dmm:(1, 257)
    generated_dict = scio.loadmat(coeff_path) # coeff_3dmm:(136, 70)

    if 'full' not in preprocess.lower():
        source_semantics = source_semantics_dict['coeff_3dmm'][:1,:70] # (1, 70)
        generated_3dmm = generated_dict['coeff_3dmm'][:,:70]
    else:
        source_semantics = source_semantics_dict['coeff_3dmm'][:1,:73] # (1, 73)，全身模式的话，还会用到第一帧图片的三个系数： scale, tx, ty，相关代码位于src/face3d/util/preprocess.py#L77和src/utils/preprocess.py#L163
        generated_3dmm = generated_dict['coeff_3dmm'][:,:70] # (136, 70)

    source_semantics_new = transform_semantic_1(source_semantics, semantic_radius) # (73, 27)
    source_semantics_ts = torch.FloatTensor(source_semantics_new).unsqueeze(0) # torch.Size([1, 73, 27])
    source_semantics_ts = source_semantics_ts.repeat(batch_size, 1, 1)
    data['source_semantics'] = source_semantics_ts

    # target 
    generated_3dmm[:, :64] = generated_3dmm[:, :64] * expression_scale # 缩放表情系数部分

    # full全身模式代表着输出的人脸可能是倾斜的，而驱动时是使用矫正过后的，驱动完要倾斜粘贴回去，相关代码还没找到
    if 'full' in preprocess.lower(): # 全身模式的话，把 scale, tx, ty这三个参数广播到每一帧
        generated_3dmm = np.concatenate([generated_3dmm, np.repeat(source_semantics[:,70:], generated_3dmm.shape[0], axis=0)], axis=1) # (136, 73)

    if still_mode: # 静止模式的话，头部运动的6+3个参数都使用第一帧画面的参数
        generated_3dmm[:, 64:] = np.repeat(source_semantics[:, 64:], generated_3dmm.shape[0], axis=0)

    # 把generated_3dmm写入txt_path+'.txt'文件
    with open(txt_path+'.txt', 'w') as f:
        for coeff in generated_3dmm:
            for i in coeff:
                f.write(str(i)[:7]   + '  '+'\t')
            f.write('\n')

    target_semantics_list = [] 
    frame_num = generated_3dmm.shape[0]
    data['frame_num'] = frame_num
    for frame_idx in range(frame_num):
        target_semantics = transform_semantic_target(generated_3dmm, frame_idx, semantic_radius)
        # print(f"==>> target_semantics.shape: {target_semantics.shape}") # (73, 27)
        target_semantics_list.append(target_semantics)

    remainder = frame_num%batch_size
    if remainder!=0:
        for _ in range(batch_size-remainder):
            target_semantics_list.append(target_semantics)

    target_semantics_np = np.array(target_semantics_list) # [frame_num, 70 or 73, semantic_radius*2+1]
    # print(f"==>> target_semantics_np.shape: {target_semantics_np.shape}") # (136, 73, 27)，可见合成每一帧画面时，会感知到前后27帧的73维姿态数据
    target_semantics_np = target_semantics_np.reshape(batch_size, -1, target_semantics_np.shape[-2], target_semantics_np.shape[-1])
    print(f"==>> target_semantics_np.shape: {target_semantics_np.shape}") # (1, 136, 73, 27)
    data['target_semantics_list'] = torch.FloatTensor(target_semantics_np)
    data['video_name'] = video_name
    data['audio_path'] = audio_path
    
    if input_yaw_list is not None:
        yaw_c_seq = gen_camera_pose(input_yaw_list, frame_num, batch_size)
        data['yaw_c_seq'] = torch.FloatTensor(yaw_c_seq)
    if input_pitch_list is not None:
        pitch_c_seq = gen_camera_pose(input_pitch_list, frame_num, batch_size)
        data['pitch_c_seq'] = torch.FloatTensor(pitch_c_seq)
    if input_roll_list is not None:
        roll_c_seq = gen_camera_pose(input_roll_list, frame_num, batch_size) 
        data['roll_c_seq'] = torch.FloatTensor(roll_c_seq)
 
    return data

def transform_semantic_1(semantic, semantic_radius):
    semantic_list =  [semantic for i in range(0, semantic_radius*2+1)] # len:27
    coeff_3dmm = np.concatenate(semantic_list, 0) # shape: (27, 73),semantic的shape是(1, 73)，这里相当于重复了27次
    return coeff_3dmm.transpose(1,0)

def transform_semantic_target(coeff_3dmm, frame_index, semantic_radius):
    num_frames = coeff_3dmm.shape[0]
    seq = list(range(frame_index- semantic_radius, frame_index + semantic_radius+1))
    index = [ min(max(item, 0), num_frames-1) for item in seq ] 
    coeff_3dmm_g = coeff_3dmm[index, :]
    return coeff_3dmm_g.transpose(1,0)

def gen_camera_pose(camera_degree_list, frame_num, batch_size):

    new_degree_list = [] 
    if len(camera_degree_list) == 1:
        for _ in range(frame_num):
            new_degree_list.append(camera_degree_list[0]) 
        remainder = frame_num%batch_size
        if remainder!=0:
            for _ in range(batch_size-remainder):
                new_degree_list.append(new_degree_list[-1])
        new_degree_np = np.array(new_degree_list).reshape(batch_size, -1) 
        return new_degree_np

    degree_sum = 0.
    for i, degree in enumerate(camera_degree_list[1:]):
        degree_sum += abs(degree-camera_degree_list[i])
    
    degree_per_frame = degree_sum/(frame_num-1)
    for i, degree in enumerate(camera_degree_list[1:]):
        degree_last = camera_degree_list[i]
        degree_step = degree_per_frame * abs(degree-degree_last)/(degree-degree_last)
        new_degree_list =  new_degree_list + list(np.arange(degree_last, degree, degree_step))
    if len(new_degree_list) > frame_num:
        new_degree_list = new_degree_list[:frame_num]
    elif len(new_degree_list) < frame_num:
        for _ in range(frame_num-len(new_degree_list)):
            new_degree_list.append(new_degree_list[-1])
    print(f"==>> len(new_degree_list): {len(new_degree_list)}")
    print(f"==>> frame_num: {frame_num}")

    remainder = frame_num%batch_size
    if remainder!=0:
        for _ in range(batch_size-remainder):
            new_degree_list.append(new_degree_list[-1])
    new_degree_np = np.array(new_degree_list).reshape(batch_size, -1) 
    return new_degree_np
    
