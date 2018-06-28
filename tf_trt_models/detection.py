from object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig
from object_detection.builders import model_builder

import os
import tarfile
import subprocess

from google.protobuf import text_format

import tensorflow as tf

from .graph_utils import convert_relu6, remove_op

input_name = 'input'
output_map = {
    'detection_scores': 'scores',
    'detection_boxes': 'boxes',
    'detection_classes': 'classes',
    'detection_masks': 'masks'
}

nets = {
    'ssd_mobilenet_v1_coco': {
        'config_url': 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config',
        'checkpoint_url': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz',
    },
    'ssd_mobilenet_v2_coco': {
        'config_url': 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config',
        'checkpoint_url': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
    },
    'ssd_inception_v2_coco': {
        'config_url': 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config',
        'checkpoint_url': 'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz',
    },
}

def download_detection_model(model, output_dir='.'):
    """Download a default detection model configuration and checkpoint.

    This function downloads a default detection model configuration and
    checkpoint.  This is only available for a subset of models in the
    TensorFlow object detection model zoo that are known to work on Jetson.

    The following models are available

    ssd_mobilenet_v1_coco
    ssd_mobilenet_v2_coco
    ssd_inception_v2_coco

    :param model: the model name from the above list
    :type model: string
    :param output_dir: the directory where files are downloaded to
    :type output_dir: string
    :return config_path:  path to the object detection pipeline config file
    :rtype string
    :return checkpoint_path:  path to the checkpoint files prefix containing trained model params
    :rtype string
    """
    global nets
    config_path = ''
    checkpoint_path = ''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    modeldir_path = os.path.join(output_dir, model)
    if not os.path.exists(modeldir_path):
        os.makedirs(modeldir_path)

    config_path = os.path.join(output_dir, model + '.config')
    if not os.path.isfile(config_path):
        subprocess.call(['wget', '--no-check-certificate', nets[model]['config_url'], '-O', config_path]) 

    modeltar_path = os.path.join(output_dir, os.path.basename(nets[model]['checkpoint_url']))
    if not os.path.isfile(modeltar_path):
        subprocess.call(['wget', '--no-check-certificate', nets[model]['checkpoint_url'], '-O', modeltar_path])

    tar_file = tarfile.open(modeltar_path)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'model.ckpt' in file_name:
            file.name = file_name
            tar_file.extract(file, modeldir_path)

    checkpoint_path = os.path.join(modeldir_path, 'model.ckpt')

    return config_path, checkpoint_path

def build_detection_graph(config, checkpoint):
    """Build an object detection model from the TensorFlow model zoo.

    This function creates an object detection model, sourced from the
    TensorFlow object detection API.

    It is necessary to use this function to generate a frozen graph that is
    compatible with TensorFlow/TensorRT integration.  In addition to generating
    a graph that is compatible with TensorFlow's TensorRT package, this
    function performs other graph modifications, such as forced device
    placement, that improve performance on Jetson.  These graph modifications
    are tested with a subset of the object detection API and may or may not
    work well with models not listed.

    The workflow when using this method is:

    1. Train model using TensorFlow object detection API
    2. Build graph configured for Jetson using this function
    3. Optimize the graph output by this method with the TensorRT package in
       TensorFlow
    4. Execute in regular TensorFlow, or using the high level TFModel class

    :param config: path to the object detection pipeline config file
    :type config: string
    :param checkpoint: path to the checkpoint files prefix containing trained model params
    :type checkpoint: string
    :returns: the configured frozen graph representing object detection model
    :rtype: a tensorflow GraphDef
    """
    global input_name, output_map

    if isinstance(config, str):
        with open(config, 'r') as f:
            config_str = f.read()
            config = TrainEvalPipelineConfig()
            text_format.Merge(config_str, config)


    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:

            model = model_builder.build(model_config=config.model, is_training=False)

            tf_input = tf.placeholder(tf.float32, [1, None, None, 3], name=input_name)
            tf_preprocessed, tf_true_image_shapes = model.preprocess(tf_input)
            tf_predictions = model.predict(preprocessed_inputs=tf_preprocessed,
                true_image_shapes=tf_true_image_shapes)
            tf_postprocessed = model.postprocess(
                prediction_dict=tf_predictions,
                true_image_shapes=tf_true_image_shapes
            )

            tf_saver = tf.train.Saver()
            tf_saver.restore(save_path=checkpoint, sess=tf_sess)

            outputs = {}
            for key, op in tf_postprocessed.items():
                if key in output_map.keys():
                    outputs[output_map[key]] = \
                        tf.identity(op, name=output_map[key])

            frozen_graph = tf.graph_util.convert_variables_to_constants(
                tf_sess,
                tf_sess.graph_def,
                output_node_names=list(outputs.keys())
            )

            frozen_graph = convert_relu6(frozen_graph)

            remove_op(frozen_graph, 'Assert')

            # force CPU device placement for NMS ops
            for node in frozen_graph.node:
                if 'NonMaxSuppression' in node.name:
                    node.device = '/device:CPU:0'

    return frozen_graph, [input_name], list(outputs.keys())
