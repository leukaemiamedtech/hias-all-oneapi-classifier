#!/bin/bash

FMSG="- Acute Lymphoblastic Leukemia oneAPI Classifier installation terminated"

read -p "? This script will install the required Python libraries. Are you ready (y/n)? " cmsg

if [ "$cmsg" = "Y" -o "$cmsg" = "y" ]; then

    echo "- Installing required Python libraries"

    sudo apt update
    sudo apt upgrade
    sudo apt install -y libgtk-3-dev
    sudo apt install gfortran
    sudo apt install libopenjp2-7
    sudo apt install libhdf5-dev libc-ares-dev libeigen3-dev
    sudo apt install libatlas-base-dev libopenblas-dev libblas-dev
    sudo apt install liblapack-dev cython
    sudo pip3 install scikit-image
    sudo pip3 install h5py
    sudo pip3 install opencv-python
    sudo pip3 install psutil
    sudo pip3 install flask
    sudo pip3 install requests
    sudo pip3 install numpy
    sudo pip3 install jsonpickle
    sudo pip3 install paho-mqtt
    sudo pip3 install pybind11
    #wget https://github.com/Qengineering/Tensorflow-Raspberry-Pi/raw/master/tensorflow-2.1.0-cp37-cp37m-linux_armv7l.whl
    #sudo -H pip3 install tensorflow-2.1.0-cp37-cp37m-linux_armv7l.whl

else
    echo $FMSG;
    exit
fi