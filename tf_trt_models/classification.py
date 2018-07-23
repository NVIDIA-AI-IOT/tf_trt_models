from collections import namedtuple

from .graph_utils import convert_relu6

import nets
import nets.inception
import nets.mobilenet_v1
import nets.resnet_v1
import nets.resnet_v2
import nets.vgg

import os
import subprocess
import tarfile

import tensorflow as tf
import tensorflow.contrib.slim as slim

NetDef = namedtuple('NetDef', ['model', 'arg_scope', 'input_width',
                    'input_height', 'preprocess', 'postprocess', 'url', 'checkpoint_name',
                    'num_classes'])


def _mobilenet_v1_1p0_224(*args, **kwargs):
    kwargs['depth_multiplier'] = 1.0
    return nets.mobilenet_v1.mobilenet_v1(*args, **kwargs)


def _mobilenet_v1_0p5_160(*args, **kwargs):
    kwargs['depth_multiplier'] = 0.5
    return nets.mobilenet_v1.mobilenet_v1(*args, **kwargs)


def _mobilenet_v1_0p25_128(*args, **kwargs):
    kwargs['depth_multiplier'] = 0.25
    return nets.mobilenet_v1.mobilenet_v1(*args, **kwargs)


def _preprocess_vgg(x):
    tf_x_float = tf.cast(x, tf.float32)
    tf_mean = tf.constant([123.68, 116.78, 103.94], tf.float32)
    return tf.subtract(tf_x_float, tf_mean)


def _preprocess_inception(x):
    tf_x_float = tf.cast(x, tf.float32)
    return 2.0 * (tf_x_float / 255.0 - 0.5)


input_name = 'input'
output_name = 'scores'
NETS = {
    'mobilenet_v1_0p25_128':
    NetDef(_mobilenet_v1_0p25_128,
           nets.mobilenet_v1.mobilenet_v1_arg_scope, 128, 128,
           _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz',
           'mobilenet_v1_0.25_128.ckpt', 1001),
    'mobilenet_v1_0p5_160':
    NetDef(_mobilenet_v1_0p5_160, nets.mobilenet_v1.mobilenet_v1_arg_scope,
           160, 160, _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz',
           'mobilenet_v1_0.5_160.ckpt', 1001),
    'mobilenet_v1_1p0_224':
    NetDef(_mobilenet_v1_1p0_224, nets.mobilenet_v1.mobilenet_v1_arg_scope,
           224, 224, _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz',
           'mobilenet_v1_1.0_224.ckpt', 1001),
    'vgg_16':
    NetDef(nets.vgg.vgg_16, nets.vgg.vgg_arg_scope, 224, 224,
           _preprocess_vgg, tf.nn.softmax,
           'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
           'vgg_16.ckpt', 1000),
    'vgg_19':
    NetDef(nets.vgg.vgg_19, nets.vgg.vgg_arg_scope, 224, 224,
           _preprocess_vgg, tf.nn.softmax,
           'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
           'vgg_19.ckpt', 1000),
    'inception_v1':
    NetDef(nets.inception.inception_v1, nets.inception.inception_v1_arg_scope,
           224, 224, _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
           'inception_v1.ckpt', 1001),
    'inception_v2':
    NetDef(nets.inception.inception_v2, nets.inception.inception_v2_arg_scope,
           224, 224, _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz',
           'inception_v2.ckpt', 1001),
    'inception_v3':
    NetDef(nets.inception.inception_v3, nets.inception.inception_v3_arg_scope,
           299, 299, _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
           'inception_v3.ckpt', 1001),
    'inception_v4':
    NetDef(nets.inception.inception_v4, nets.inception.inception_v4_arg_scope,
           299, 299, _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
           'inception_v4.ckpt', 1001),
    'inception_resnet_v2':
    NetDef(nets.inception.inception_resnet_v2,
           nets.inception.inception_resnet_v2_arg_scope, 299, 299,
           _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz',
           'inception_resnet_v2_2016_08_30.ckpt', 1001),
    'resnet_v1_50':
    NetDef(nets.resnet_v1.resnet_v1_50, nets.resnet_v1.resnet_arg_scope,
           224, 224, _preprocess_vgg, tf.nn.softmax,
           'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
           'resnet_v1_50.ckpt', 1000),
    'resnet_v1_101':
    NetDef(nets.resnet_v1.resnet_v1_101, nets.resnet_v1.resnet_arg_scope,
           224, 224, _preprocess_vgg, tf.nn.softmax,
           'http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
           'resnet_v1_101.ckpt', 1000),
    'resnet_v1_152':
    NetDef(nets.resnet_v1.resnet_v1_152, nets.resnet_v1.resnet_arg_scope,
           224, 224, _preprocess_vgg, tf.nn.softmax,
           'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',
           'resnet_v1_152.ckpt', 1000),
    'resnet_v2_50':
    NetDef(nets.resnet_v2.resnet_v2_50, nets.resnet_v2.resnet_arg_scope,
           299, 299, _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
           'resnet_v2_50.ckpt', 1001),
    'resnet_v2_101':
    NetDef(nets.resnet_v2.resnet_v2_101, nets.resnet_v2.resnet_arg_scope,
           299, 299, _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
           'resnet_v2_101.ckpt', 1001),
    'resnet_v2_152':
    NetDef(nets.resnet_v2.resnet_v2_152, nets.resnet_v2.resnet_arg_scope,
           299, 299, _preprocess_inception, tf.nn.softmax,
           'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
           'resnet_v2_152.ckpt', 1001),
}


def download_classification_checkpoint(model, output_dir='.'):
    """Downloads an image classification model pretrained checkpoint by name

    :param model: the model name (see table)
    :type model: string
    :param output_dir: the directory where files are downloaded to
    :type output_dir: string
    :return checkpoint_path:  path to the checkpoint file containing trained model params
    :rtype string
    """
    global NETS, input_name, output_name
    checkpoint_path = ''

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    modeldir_path = os.path.join(output_dir, model)
    if not os.path.exists(modeldir_path):
      os.makedirs(modeldir_path)

    modeltar_path = os.path.join(output_dir, os.path.basename(NETS[model].url))
    if not os.path.isfile(modeltar_path):
        subprocess.call(['wget', '--no-check-certificate', NETS[model].url, '-O', modeltar_path])
     
    checkpoint_path = os.path.join(modeldir_path, NETS[model].checkpoint_name)
    if not os.path.isfile(checkpoint_path):
        subprocess.call(['tar', '-xzf', modeltar_path, '-C', modeldir_path])

    return checkpoint_path


def build_classification_graph(model, checkpoint, num_classes):
    """Builds an image classification model by name

    This function builds an image classification model given a model
    name, parameter checkpoint file path, and number of classes.  This
    function performs some graph processing (such as replacing relu6(x)
    operations with relu(x) - relu(x-6)) to produce a graph that is
    well optimized by the TensorRT package in TensorFlow 1.7+.

    :param model: the model name (see table)
    :type model: string
    :param checkpoint: the checkpoint file path
    :type checkpoint: string
    :param num_classes: the number of output classes
    :type num_classes: integer

    :returns: the TensorRT compatible frozen graph
    :rtype: a tensorflow.GraphDef
    """
    global NETS, input_name, output_name

    net = NETS[model]
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:

            tf_input = tf.placeholder(tf.float32, [None, net.input_height, net.input_width, 3],
                    name=input_name)
            tf_preprocessed = net.preprocess(tf_input)

            with slim.arg_scope(net.arg_scope()):
                tf_net, tf_end_points = net.model(tf_preprocessed, is_training=False,
                    num_classes=num_classes)

            tf_output = net.postprocess(tf_net, name=output_name)

            # load checkpoint
            tf_saver = tf.train.Saver()
            tf_saver.restore(save_path=checkpoint, sess=tf_sess)

            # freeze graph
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                tf_sess,
                tf_sess.graph_def,
                output_node_names=[output_name]
            )

            # remove relu 6
            frozen_graph = convert_relu6(frozen_graph)

    return frozen_graph, [input_name], [output_name]
