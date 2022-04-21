#!/usr/bin/env bash

if [ -d data/MNIST/MNIST/raw ]; then
    echo "data directory already present, exiting"
    exit 1
fi

mkdir data/MNIST/MNIST/raw
wget --recursive --level=1 --cut-dirs=3 --no-host-directories \
  --directory-prefix=data/MNIST/MNIST/raw --accept '*.gz' http://yann.lecun.com/exdb/mnist/
pushd data/MNIST/MNIST/raw
gunzip *
