import os, sys
import gradio as gr
import time
from src.utils.my_utils import *

class SadTalker():
    """
    对sadtalker modelsccope library做个简单的封装
    """
    def __init__(self):
        self.save_dir = './results/'

    def __call__(self, *args, **kwargs) -> Any:
        # two required arguments
        source_image = kwargs.get("source_image") or args[0]
        driven_audio = kwargs.get('driven_audio') or args[1]
        # other optional arguments
        kwargs = {
            'preprocess' : kwargs.get('preprocess') or args[2], 
            'still_mode' : kwargs.get('still_mode') or args[3],
            'use_enhancer' : kwargs.get('use_enhancer') or args[4],
            'batch_size' : kwargs.get('batch_size') or args[5],
            'size' : kwargs.get('size') or args[6], 
            'pose_style' : kwargs.get('pose_style') or args[7],
            'exp_scale' : kwargs.get('exp_scale') or args[8],
            'result_dir': self.save_dir
        }
        inference = pipeline('talking-head', model='wwd123/sadtalker', model_revision='v1.0.0')
        print("initialized sadtalker pipeline")
        video_path = inference(source_image, driven_audio=driven_audio, **kwargs)
        return video_path
    
def launch_pipeline_talkinghead(source_image, driven_audio, preprocess='crop', 
        still_mode=True,  use_enhancer=False, batch_size=1, size=256, 
        pose_style = 0, exp_scale=1.0):
    if not check_ffmpeg():
        raise gr.Error("请先安装ffmpeg，然后刷新网页（Please install ffmpeg, then restart the webpage）")

    if not source_image:
        raise gr.Error('请选择一张源图片(Please select 1 source image)')
    if not driven_audio:
        raise gr.Error('请上传一段wav、mp3音频(Please upload 1 wav or mp3 audio)')

    user_directory = os.path.expanduser("~")
    if not os.path.exists(os.path.join(user_directory, '.cache', 'modelscope', 'hub', 'wwd123', 'sadtalker')):
        gr.Info("第一次初始化会比较耗时，请耐心等待(The first time initialization will take time, please wait)")

    sadtalker = SadTalker()

    video = sadtalker(source_image, driven_audio, preprocess, 
                        still_mode, use_enhancer, batch_size, size, pose_style, exp_scale)

    return video

def sadtalker_webui():
    with gr.Blocks() as demo:
        gr.Markdown("""该标签页的功能基于[SadTalker](https://sadtalker.github.io)实现，要使用该标签页，请按照[教程](https://github.com/wwdok/sadtalker_modelscope/tree/master/doc/installation_CN.md)安装相关依赖。\n
                    The function of this tab is implemented based on [SadTalker](https://sadtalker.github.io), to use this tab, you should follow the installation [guide](https://github.com/wwdok/sadtalker_modelscope/tree/master/doc/installation.md) """)
        
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                source_image = gr.Image(label="源图片(source image)", source="upload", type="filepath")
                driven_audio = gr.Audio(label="驱动音频(driven audio)", source="upload", type="filepath")
                input_text = gr.Textbox(label="用文本生成音频(Generating audio from text)", lines=1, value="请在此处输入您想要合成语音的文本")
                speaker = gr.Dropdown(choices=list(tts_speakers_map.keys()), value="普通话(中国大陆)-Xiaoxiao-女", label="请根据输入文本选择对应的语言和说话人(Select speaker according the language of input text)")
                tts = gr.Button('生成音频(Generate audio)')
                tts.click(fn=text_to_speech_edge, inputs=[input_text, speaker], outputs=[driven_audio])
                                
            with gr.Column(variant='panel'): 
                with gr.Box():
                    gr.Markdown("设置(Settings)")
                    with gr.Column(variant='panel'):
                        pose_style = gr.Slider(minimum=0, maximum=45, step=1, label="头部姿态(Pose style)", info="模型自主学习到的头部姿态(the head pose style that model learn)", value=0)
                        exp_weight = gr.Slider(minimum=0.5, maximum=2, step=0.1, label="表情系数(expression scale)", info="数值越大，表情越夸张(the higher, the more exaggerated)", value=1)
                        with gr.Row():
                            size_of_image = gr.Radio([256, 512], value=256, label='人脸模型分辨率(face model resolution)', info="使用哪种输入分辨率的模型(use which model with this input size)")
                            preprocess_type = gr.Radio(['crop', 'resize','full'], value='full', label='预处理(preprocess)', info="如果源图片是全身像，`crop`会裁剪到只剩人脸区域")
                        is_still_mode = gr.Checkbox(value=True, label="静止模式(Still Mode)", info="更少的头部运动(fewer head motion)")
                        enhancer = gr.Checkbox(label="使用GFPGAN增强人脸清晰度(GFPGAN as Face enhancer)")
                        batch_size = gr.Slider(label="批次大小(batch size)", step=1, maximum=10, value=1, info="当处理长视频，可以分成多段并行合成(when systhesizing long video, this will process it in parallel)")
                        submit = gr.Button('生成(Generate)', variant='primary')
                with gr.Box():
                        gen_video = gr.Video(label="生成的视频(Generated video)", format="mp4", width=256)

        submit.click(fn=launch_pipeline_talkinghead, inputs=[source_image, driven_audio, preprocess_type,
                    is_still_mode, enhancer, batch_size, size_of_image, pose_style, exp_weight], 
                    outputs=[gen_video])
        with gr.Row():
            examples = [
                [   f'examples/source_image/man.png',
                    f'examples/driven_audio/chinese_poem1.wav',
                    'full',
                    True,
                    False],
                [   f'examples/source_image/women.png',
                    f'examples/driven_audio/chinese_poem2.wav',
                    'full',
                    True,
                    False],
            ]
            gr.Examples(examples=examples, inputs=[source_image, driven_audio, preprocess_type, is_still_mode, enhancer], 
                        outputs=[gen_video],  fn=launch_pipeline_talkinghead, cache_examples=os.getenv('SYSTEM') == 'spaces')

    return demo
 

if __name__ == "__main__":
    demo = sadtalker_webui()
    demo.queue()
    demo.launch()
