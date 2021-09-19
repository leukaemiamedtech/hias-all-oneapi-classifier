#!/bin/bash

FMSG="HIAS ALL oneAPI Classifier installation terminated!"

printf -- 'This script will install HIAS ALL oneAPI Classifier on your Ubuntu development machine.\n';
printf -- '\033[33m WARNING: This is an inteteractive installation, please follow instructions provided. \033[0m\n';

read -p "Proceed (y/n)? " proceed
if [ "$proceed" = "Y" -o "$proceed" = "y" ]; then
    conda install flask
    conda install jsonpickle
    conda install jupyter
    conda install matplotlib
    conda install nb_conda
    conda install opencv
    conda install -c conda-forge paho-mqtt
    conda install Pillow
    conda install psutil
    conda install requests
    conda install scikit-learn
    conda install scikit-learn-intelex
    conda install scikit-image
    conda install tornado
    pip install mlxtend
    printf -- '\033[32m SUCCESS: Congratulations! HIAS ALL oneAPI Classifier installed successfully! \033[0m\n';
else
    echo $FMSG;
    exit 1
fi