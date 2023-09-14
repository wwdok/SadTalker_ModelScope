# Copyright (c) Alibaba, Inc. and its affiliates.

import torch

from modelscope.models.base import TorchModel
from modelscope.preprocessors.base import Preprocessor
from modelscope.pipelines.base import Model, Pipeline
from modelscope.utils.config import Config
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.models.builder import MODELS
# custom import
from src.gradio_demo import SadTalker
from modelscope.utils.logger import get_logger
import os

logger = get_logger()

@MODELS.register_module('talking-head', module_name='my-custom-model')
class MyCustomModel(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.model = self.init_model(**kwargs)

    def forward(self, source_image, **forward_params):
        print(f"==>> source_image: {source_image}")
        print(f"==>> sadtalker forward_params: {forward_params}")
        return self.model.test(source_image, **forward_params)

    def init_model(self, **kwargs):
        """Provide default implementation based on TorchModel and user can reimplement it.
            include init model and load ckpt from the model_dir, maybe include preprocessor
            if nothing to do, then return lambda x: x
        """
        if not (os.path.exists('checkpoints') and os.path.exists('gfpgan')):
            print("Download sadtalker needed models...")
            os.system('bash download_models.sh')
        model = SadTalker(checkpoint_path='checkpoints', config_path='src/config')
        logger.info("Initialized SadTalker")
        return model


@PREPROCESSORS.register_module('talking-head', module_name='my-custom-preprocessor')
class MyCustomPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainsforms = self.init_preprocessor(**kwargs)

    def __call__(self, results):
        return self.trainsforms(results)

    def init_preprocessor(self, **kwarg):
        """ Provide default implementation based on preprocess_cfg and user can reimplement it.
            if nothing to do, then return lambda x: x
        """
        return lambda x: x


@PIPELINES.register_module('talking-head', module_name='my-custom-pipeline')
class MyCustomPipeline(Pipeline):
    """ Give simple introduction to this pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> input = "Hello, ModelScope!"
    >>> my_pipeline = pipeline('my-task', 'my-model-id')
    >>> result = my_pipeline(input)

    """

    def __init__(self, model, preprocessor=None, **kwargs):
        """
        use `model` and `preprocessor` to create a custom pipeline for prediction
        Args:
            model: model id on modelscope hub.
            preprocessor: the class of method be init_preprocessor
        """
        super().__init__(model=model, auto_collate=False)
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or Model'
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        pipe_model.eval()

        if preprocessor is None:
            preprocessor = MyCustomPreprocessor()
        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output

        Default Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, pipeline_parameters, {}

    def _check_input(self, inputs):
        pass

    def _check_output(self, outputs):
        pass

    def forward(self, source_image, **forward_params):
        """ Provide default implementation using self.model and user can reimplement it
        """
        return super().forward(source_image, **forward_params)

    def postprocess(self, inputs):
        """ If current pipeline support model reuse, common postprocess
            code should be write here.

        Args:
            inputs:  input data

        Return:
            dict of results:  a dict containing outputs of model, each
                output should have the standard output name.
        """
        return inputs


# Tips: usr_config_path is the temporary save configuration location， after upload modelscope hub, it is the model_id
usr_config_path = '/tmp/snapdown/'
config = Config({
    "framework": 'pytorch',
    "task": 'talking-head',
    "model": {'type': 'my-custom-model'},
    "pipeline": {"type": "my-custom-pipeline"},
    "allow_remote": True
})
config.dump('/tmp/snapdown/' + 'configuration.json')

if __name__ == "__main__":
    from modelscope.models import Model
    from modelscope.pipelines import pipeline
    # model = Model.from_pretrained(usr_config_path)
    source_image = 'examples/source_image/man.png'
    driven_audio = 'examples/driven_audio/chinese_poem1.wav'
    inference = pipeline('talking-head', model=usr_config_path)
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