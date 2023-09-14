---
tasks:
- talking-head
widgets:
    - task: talking-head
domain:
- cv
frameworks:
- pytorch
backbone:
- encoder-decoder
customized-quickstart: True
license: Apache License 2.0
tags:
- sadtalker
- talking head
- 数字人

---

# SadTalker
本仓库是基于 https://github.com/OpenTalker/SadTalker （ed419f275f8a5cae7ca786349787ffebce5bd59e）改编而来。
不同点主要体现在：
* 将代码仓库打包成modelscope library，这样就能便用几行代码调用sadtalker的能力，方便集成到其他项目里
* 支持使用文本生成语音(仅限Linnux系统)


该仓库包含两个代码入口，一个是本地部署的、以gradio_app.py为入口，一个是通过modelscope调用、以ms_wrapper.py为入口。
如果你是想要使用第一种方式，请参考下面的安装，如果你是想使用第二种方式，请参考下面的代码范例。

# 安装
以Linux为例，考虑到在安装过程中可能会出现某些pypi包会覆盖安装的问题，安装上讲究顺序。
1. 如果你还没安装pytorch，先安装pytorch：注意pytorch版本要跟cuda版本对应。
2. 首先安装kantts:`pip install kantts -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html`, kantts会重装最新版的numpy，这会导致错误，所以下面我们在requirements.txt里重新安装较低版的numpy
3. 然后安装其他pypi包：`pip install -r requirements.txt`
4. 下面是原SadTalker里没有但我发现我本机需要安装的:
```
tb-nightly
pytorch_wavelets
tensorboardX
typing-extensions==4.3.0
```
5. 上面的安装过程中很可能会重装opencv python，而我的服务器没有GUI，所以我需要再重装一下opencv python headless，如果你的电脑有GUI，则跳过这一步
* 安装ffmpeg。首先通过`ffmpeg -version`检查你的电脑是否已经安装过ffmpeg，没有的话，通过以下两行命令安装：
```
sudo apt update
sudo apt install ffmpeg
```
windows安装ffmpeg会有所不同，请百度一下。
* 下载预训练模型权重：`bash download_models.sh`。这一步会可能耗时，你也可以把里面的链接粘贴到其他下载器下载，下载完再移动到指定文件夹下。完成后会生成checkpoints和gfpgan文件夹，里面分别有4个模型权重文件。
* 运行`python gradio_app.py`启动web ui。


# 代码范例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
import shutil

source_image = 'examples/source_image/man.png'
driven_audio = 'examples/driven_audio/chinese_poem1.wav'
out_dir = 'your-desired_output_directory'
inference = pipeline('talking-head', model='wwdok/sadtalker', model_revision='v1.0.0')
# custom arguments
kwargs = {
    'preprocess' : 'full', # 'crop', 'resize','full'
    'still_mode' : True,
    'use_enhancer' : False,
    'batch_size' : 1,
    'size' : 256, # 256, 512
    'pose_style' : 0,
    'exp_scale' : 1,
    'result_dir': './results/'
}

video_path = inference(source_image, driven_audio=driven_audio, **kwargs)
print(video_path)
shutil.move(video_path, out_dir)
```

## 参数说明
* `source_image`: 必填，要驱动的人脸图片的路径。
* `driven_audio`: 必填，且必须带上`driven_audio=`，驱动音频文件的路径，支持wav,mp3格式。
* `preprocess`：full：输出的视频帧跟原图一样大，crop：输出的视频帧只有裁剪后的人脸区域。
* `still_mode`: 设置为True会减少头部运动。
* `use_enhancer`: 是否使用GFPGAN对人脸增强，即增加清晰度。
* `batch_size`: 该值代表了Face Renderer阶段并行处理的批次数，因为这一阶段是最耗时的。比如batch size=1时，Face Renderer需要100个时间步，batch size=10时，Face Renderer仅需要10个时间步，但是batch size增大有两个问题，第一，GPU显存占用增大，第二，预处理会占用时间，只有当需要合成的视频比较长时，增大batch size才有用。
* `size`: 人脸裁剪成的大小。
* `pose_style`: 是条件VAE（即PoseVAE）的条件输入，使用的地方最终位于src/audio2pose_models/cvae.py里的`class DECODER`的`def forward`。
* `exp_scale`: 越大的话表情越夸张。
* `result_dir`: 结果输出路径。

# TODO
* 将模型权重放到国内，加速国内用户下载
* 支持从其他标签页加载图像（用于集成到facechain）
* 支持将sadtalker合成的视频再喂给wav2lip，优化唇部