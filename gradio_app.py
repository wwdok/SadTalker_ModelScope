import os, sys
import gradio as gr
# import tempfile
# from huggingface_hub import snapshot_download
# from modelscope.outputs import OutputKeys
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from src.gradio_demo import SadTalker
from src.utils.text2speech import text_to_speech_edge, tts_speakers_map

def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)
    gr.Markdown("""该标签页的功能基于[SadTalker](https://sadtalker.github.io)实现，要使用该标签页，请按照README安装相关依赖。\n
            The function of this tab is implemented based on [SadTalker](https://sadtalker.github.io), to use this tab, you should follow the installation guide in README. """)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")
        
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                source_image = gr.Image(label="源图片(source image)", source="upload", type="filepath")
                driven_audio = gr.Audio(label="驱动音频(driven audio)", source="upload", type="filepath")
                input_text = gr.Textbox(label="用文本生成音频(Generating audio from text)", lines=1, value="大家好，欢迎使用阿里达摩院开源的face chain项目！")
                speaker = gr.Dropdown(choices=list(tts_speakers_map.keys()), value="普通话(中国大陆)-Xiaoxiao-女", label="请根据输入文本选择对应的语言和说话人(Select speaker according the language of input text)")
                tts = gr.Button('生成音频(Generate audio)')
                tts.click(fn=text_to_speech_edge, inputs=[input_text, speaker], outputs=[driven_audio])

                        # if sys.platform == 'linux': 
                        #     model_id = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
                        #     tts_talker = pipeline(task=Tasks.text_to_speech, model=model_id)
                        #     with gr.Column(variant='panel'):
                        #         input_text = gr.Textbox(label="Generating audio from text", lines=5, placeholder="please enter some text here, we genreate the audio from text.")
                        #         d = {"男生": "zhiyan_emo", "女生": "zhitian_emo"}
                        #         speaker = gr.Dropdown(d.keys(), value="女生", label="Select speaker")
                        #         tts = gr.Button('Generate audio',elem_id="sadtalker_audio_generate", variant='primary')
                        #         lambda_fn = lambda input_text, speaker: tts_talker(input_text, voice=d[speaker])[OutputKeys.OUTPUT_WAV]
                        #         tts.click(fn=lambda_fn, inputs=[input_text, speaker], outputs=[driven_audio])
                                               
            with gr.Column(variant='panel'): 
                with gr.Box():
                    gr.Markdown("设置(Settings)")
                    with gr.Column(variant='panel'):
                    # with gr.Accordion("高级选项(Advanced Options)", open=False):
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
                        infer_progress = gr.Textbox(value="当前无任务(No task currently)", show_label=False, interactive=False)
                        gen_video = gr.Video(label="Generated video", format="mp4", width=256)

        submit.click(
                    fn=sad_talker.test, 
                    inputs=[source_image,
                            driven_audio,
                            preprocess_type,
                            is_still_mode,
                            enhancer,
                            batch_size,                            
                            size_of_image,
                            pose_style,
                            exp_weight
                            ], 
                    outputs=[gen_video]
                    )
        with gr.Row():
            examples = [
                [
                    'examples/source_image/man.png',
                    'examples/driven_audio/chinese_poem1.wav',
                    'full',
                    True,
                    False
                ],
                [
                    'examples/source_image/women.png',
                    'examples/driven_audio/chinese_poem2.wav',
                    'full',
                    False,
                    False
                ],
            ]
            gr.Examples(examples=examples,
                        inputs=[
                            source_image,
                            driven_audio,
                            preprocess_type,
                            is_still_mode,
                            enhancer], 
                        outputs=[gen_video],
                        fn=sad_talker.test,
                        cache_examples=os.getenv('SYSTEM') == 'spaces')

    return sadtalker_interface
 

if __name__ == "__main__":
    demo = sadtalker_demo()
    demo.queue()
    demo.launch()
