#!/bin/bash

INSTALL_PROTOC=$PWD/scripts/install_protoc.sh
MODELS_DIR=$PWD/third_party/models

PYTHON=python

if [ $# -eq 1 ]; then
  PYTHON=$1
fi

echo $PYTHON

# install protoc
echo "Downloading protoc"
source $INSTALL_PROTOC
PROTOC=$PWD/data/protoc/bin/protoc

# install tensorflow models
git submodule update --init

pushd $MODELS_DIR/research
echo $PWD
echo "Installing object detection library"
echo $PROTOC
$PROTOC object_detection/protos/*.proto --python_out=.
$PYTHON setup.py install --user
popd

pushd $MODELS_DIR/research/slim
echo $PWD
echo "Installing slim library"
$PYTHON setup.py install --user
popd

echo "Installing tf_trt_models"
echo $PWD
$PYTHON setup.py install --user
