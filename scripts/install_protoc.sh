#!/bin/bash

BASE_URL="https://github.com/google/protobuf/releases/download/v3.5.1/"

mkdir -p data/protoc
cd data/protoc
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ] ; then
  filename="protoc-3.5.1-linux-aarch_64.zip"
elif [ "$ARCH" == "x86_64" ] ; then
  filename="protoc-3.5.1-linux-x86_64.zip"
else
  echo ERROR: $ARCH not supported.
  exit 1;
fi
wget --no-check-certificate ${BASE_URL}${filename}
unzip ${filename}
sudo mv bin/protoc /usr/bin/protoc
sudo mv include/google /usr/local/include/google
cd ../..
rm -rf data/protoc
