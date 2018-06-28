#!/bin/bash

ROOT_DIR=$PWD
THIRD_PARTY_DIR=$ROOT_DIR/third_party
MODELS_DIR=$THIRD_PARTY_DIR/models

PYTHON=python

if [ $# -eq 1 ]; then
  PYTHON=$1
fi

echo $PYTHON

# install protoc
(
source scripts/install_protoc.sh
)

# install tensorflow models
(
git submodule update --init
cd $MODELS_DIR
cd research
protoc object_detection/protos/*.proto --python_out=.
sudo $PYTHON setup.py install
cd slim
sudo $PYTHON setup.py install
cd $ROOT_DIR
)

# install this project
(
sudo $PYTHON setup.py install
)
