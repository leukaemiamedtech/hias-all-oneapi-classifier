#!/bin/bash

echo "-- Installing requirements"
echo " "
conda install opencv
conda install psutil
conda install requests
conda install flask
conda install matplotlib
conda install tornado
conda install Pillow
conda install jsonpickle
conda install scikit-learn
conda install scikit-image
pip install mlxtend
echo "-- Done"