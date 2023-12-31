 # SadTalker within modelscope

This repository is adapted from: https://github.com/OpenTalker/SadTalker (ed419f275f8a5cae7ca786349787ffebce5bd59e), The goal is to wrap sadtalker as a modelscope library, so that sadtalker's capabilities can be invoked with a few lines of code for easy integration into other projects, and this repo supports using TTS (edge tts and sambert) so that users don't need to upload their own audio. Easy for users to play quickly.

Modelscope hosted repo: https://modelscope.cn/models/wwd123/sadtalker ( it's the mini edition of this github repo, their codes have slight differences.)

The repo can mainly divided into two ways to use, the first one running the code in the root directory, the entrance file is `gradio_app.py`. The other one running the code in the modelscope cache directory, the entrance file is `gradio_app_ms.py`, `demo.ipynb`, `ms_wrapper.py`. 

# Usage 1

## Setup environment

Taking Linux as an example, the installation order is important due to the possibility of certain pypi packages overwriting each other during the installation process.

1. If you haven't installed PyTorch yet, install it first. Make sure the PyTorch version matches your CUDA version.
2. Install `kantts`: `pip install kantts -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html`. Note that kantts will reinstall the latest version of numpy, which can cause errors. That's why we reinstall a lower version of numpy in the `requirements.txt`. This dependency is installed for using sambert TTS. If you don't need sambert TTS, you can skip installing it.
3. Then install other pypi packages: `pip install -r requirements.txt`.
4. Here are some additional packages that are not included in the original SadTalker repository but are required on my local machine:
```
tb-nightly
pytorch_wavelets
tensorboardX
typing-extensions==4.3.0
```
5. During the installation process mentioned above, OpenCV Python is likely to be reinstalled. Since my server doesn't have a GUI, I need to reinstall OpenCV Python headless. If your computer has a GUI, you can skip this step.
6. Install FFmpeg. First, check if FFmpeg is already installed on your computer by running `ffmpeg -version`. If it is not installed, use the following commands to install it:
```
sudo apt update
sudo apt install ffmpeg
```
Installing FFmpeg on Windows will be different, so please search online for the instructions.
7. Download the pretrained model weights: `bash download_models.sh`. This step may take some time. Alternatively, you can copy the links inside the script and use a download manager to download them. Once downloaded, move the weights to the specified folders. After completion, the `checkpoints` and `gfpgan/weights` folders will be generated, each containing four model weight files.

## Running

* Run `python gradio_app.py` to start the web UI.


# Usage 2

## Setup environment

See[this doc](doc/installation.md)

## Running

```python
from modelscope.pipelines import pipeline

inference = pipeline('talking-head', model='wwd123/sadtalker', model_revision='v1.0.0') # Please use the latest model_revision
# Two mandatory parameters
source_image = 'examples/source_image/man.png' # Modify this to your actual path
driven_audio = 'examples/driven_audio/chinese_poem1.wav' # Modify this to your actual path
# Other optional parameters
out_dir = './results/' # Output folder
kwargs = {
    'preprocess' : 'full', # 'crop', 'resize', 'full'
    'still_mode' : True,
    'use_enhancer' : False,
    'batch_size' : 1,
    'size' : 256, # 256, 512
    'pose_style' : 0,
    'exp_scale' : 1,
    'result_dir': out_dir
}

video_path = inference(source_image, driven_audio=driven_audio, **kwargs)
print(f"==>> video_path: {video_path}")
```

This repo also provides the webui that invokes the wrapped sadtalker modelscope library：`python gradio_app_ms.py`.

You can try it out on Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1C2TjndoDsUXlW6P10peN66p4I9ImEHyt?usp=sharing/), or run it locally with [demo.ipynb](demo.ipynb).

### Parameter Explanation

* `source_image`: Required. The path to the source face image to be driven.
* `driven_audio`: Required and must be provided as `driven_audio=`, the path to the audio file for driving, supporting WAV and MP3 formats.
* `preprocess`: Options are `full` (output video frames with the same size as the original image) or `crop` (output video frames with only the cropped face region).
* `still_mode`: Set to `True` to reduce head movement.
* `use_enhancer`: Whether to use GFPGAN for face enhancement, increasing clarity.
* `batch_size`: The number of batches processed in parallel during the Face Renderer stage. This stage is the most time-consuming. For example, when `batch_size=1`, the Face Renderer requires 100 time steps, but when `batch_size=10`, the Face Renderer only needs 10 time steps. However, increasing the batch size has two issues: increased GPU memory usage and preprocessing time. Increasing the batch size is only useful when generating longer videos.
* `size`: The size to which the face is cropped.
* `pose_style`: The conditional input for the conditional VAE (PoseVAE), used in `class DECODER` of `src/audio2pose_models/cvae.py`.
* `exp_scale`: The larger the value, the more exaggerated the expression.
* `result_dir`: The path for the output results.

