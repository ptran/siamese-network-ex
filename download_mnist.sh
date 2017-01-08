#!/bin/sh

if [ ! -d $PWD/data ]; then
    echo "Creating the $PWD/data directory..."
    mkdir data
fi

echo "Downloading MNIST dataset to $PWD/data..."
wget -P $PWD/data http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -P $PWD/data http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -P $PWD/data http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -P $PWD/data http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
cd data
find . -name "*.gz" -exec gzip -d {} \;
cd ..
echo "MNIST downloaded."
