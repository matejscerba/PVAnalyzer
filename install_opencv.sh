#!/bin/bash
mkdir -p opencv && cd opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.1.zip
unzip opencv.zip
unzip opencv_contrib.zip
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.5.1/modules ../opencv-4.5.1 -DCMAKE_INSTALL_PREFIX=$HOME/.local || { echo 'Configuring failed.'; exit 1; }
cmake --build . || { echo 'Build failed.'; exit 1; }
make install