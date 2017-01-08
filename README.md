Siamese Network Example
=======================

This repo showcases how to create a Siamese network using the tools provided by
the [*dlib* machine learning library](http://dlib.net/)
([github link](https://github.com/davisking/dlib)).

![Embedding Output](embedding.png)

Requirements
------------

### dlib
* ***Version:*** >19.0
* **Dependencies**
  * a `C++11`-compatible compiler (`g++`, `clang++`, etc...)
  * `CUDA 7.5`
  * `cuDNN v5`

### CMake
* ***Version:*** 2.6

Build
-----

In order to build this project, run the following commands at this repo's root
directory.

### Linux
``` bash
    # create a directory to contain all build by-products
    mkdir build
    cd build
    cmake -DDLIB_CMAKE_FILE=$DLIB_ROOT/dlib/cmake ..
    make && make install
```
`$DLIB_ROOT` is the path to the root directory of the *dlib* library.


### Windows
``` bash
    mkdir build
    cd build
    cmake -DDLIB_CMAKE_FILE=$DLIB_ROOT\dlib\cmake ..
    cmake --build . --config release --target install
```
