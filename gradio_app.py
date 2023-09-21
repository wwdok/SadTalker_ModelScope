import os, sys
import gradio as gr
import tempfile
from huggingface_hub import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from src.gradio_demo import SadTalker


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
    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")
        
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", source="upload", type="filepath", elem_id="img2img_image").style(width=512)
                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload OR TTS'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio", source="upload", type="filepath")

                        if sys.platform == 'linux': 
                            model_id = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
                            tts_talker = pipeline(task=Tasks.text_to_speech, model=model_id)
                            with gr.Column(variant='panel'):
                                input_text = gr.Textbox(label="Generating audio from text", lines=5, placeholder="please enter some text here, we genreate the audio from text.")
                                d = {"男生": "zhiyan_emo", "女生": "zhitian_emo"}
                                speaker = gr.Dropdown(d.keys(), value="女生", label="Select speaker")
                                tts = gr.Button('Generate audio',elem_id="sadtalker_audio_generate", variant='primary')
                                lambda_fn = lambda input_text, speaker: tts_talker(input_text, voice=d[speaker])[OutputKeys.OUTPUT_WAV]
                                tts.click(fn=lambda_fn, inputs=[input_text, speaker], outputs=[driven_audio])
                                               
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        gr.Markdown("need help? please visit our [best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) for more detials")
                        with gr.Column(variant='panel'):
                        # with gr.Accordion("高级选项(Advanced Options)", open=False):
                            pose_style = gr.Slider(minimum=0, maximum=45, step=1, label="Pose style", value=0) # 
                            exp_weight = gr.Slider(minimum=0.5, maximum=2, step=0.1, label="expression scale", value=1)
                            size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?") # 
                            preprocess_type = gr.Radio(['crop', 'resize','full'], value='full', label='preprocess', info="How to handle input image?")
                            is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                            batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=1, info="You can set this larger when generating long video")
                            enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')
                            
                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)

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
