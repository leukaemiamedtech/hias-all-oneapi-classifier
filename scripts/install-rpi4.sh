#!/bin/bash

FMSG="HIAS ALL oneAPI Classifier installation terminated!"

printf -- 'This script will install HIAS ALL oneAPI Classifier on your Raspberry Pi.\n';
printf -- '\033[33m WARNING: This is an inteteractive installation, please follow instructions provided. \033[0m\n';

read -p "Proceed (y/n)? " proceed
if [ "$proceed" = "Y" -o "$proceed" = "y" ]; then
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
    printf -- '\033[32m SUCCESS: Congratulations! HIAS ALL oneAPI Classifier installed successfully! \033[0m\n';
else
    echo $FMSG;
    exit 1
fi