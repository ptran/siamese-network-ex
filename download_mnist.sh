#!/bin/sh

echo "Downloading MNIST dataset to $PWD/data..."
wget -P data/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -P data/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -P data/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -P data/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
cd data/
find . -name "*.gz" -exec gzip -d {} \;
cd ..
echo "MNIST downloaded."
