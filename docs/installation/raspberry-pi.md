# Installation (Raspberry Pi)

![Acute Lymphoblastic Leukemia oneAPI Classifier](../img/project-banner.jpg)

# Introduction
This guide will guide you through the installation process for the **HIAS Acute Lymphoblastic Leukemia oneAPI Classifier** on your Ubuntu development machine.

&nbsp;

# Raspberry Pi OS
For this Project, the operating system choice is [Raspberry Pi OS](https://www.raspberrypi.org/downloads/raspberry-pi-os/ "Raspberry Pi OS").

&nbsp;

# Intel® Distribution of OpenVINO™ Toolkit
To install Intel® Distribution of OpenVINO™ Toolkit for Raspberry Pi, navigate to the home directory on your Raspberry Pi and use the following commands:

```
  wget https://download.01.org/opencv/2021/openvinotoolkit/2021.2/l_openvino_toolkit_runtime_raspbian_p_2021.2.185.tgz
```
```
  sudo mkdir -p /opt/intel/openvino
  sudo tar -xf  l_openvino_toolkit_runtime_raspbian_p_2021.2.185.tgz  --strip 1 -C /opt/intel/openvino
```
```
  sudo apt install cmake
  source /opt/intel/openvino/bin/setupvars.sh
  echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc
```

&nbsp;

# Intel® Movidius™ Neural Compute Stick 2
Now we will set up ready for Neural Compute Stick 2.
```
  sudo usermod -a -G users "$(whoami)"
```
Now close your existing terminal and open a new open. Once in your new terminal use the following commands:
```
  sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh
```

&nbsp;

# Transfer Files

Next you need to transfer the project folder to your Raspberry Pi, make sure that you have all of the files from the model directory.

&nbsp;

# Software Install

All other requirements are included in **scripts/install-rpi4.sh**. You can run this file on machine by navigating to the project root in terminal and using the commands below:

```
 sh scripts/install-rpi4.sh
```

&nbsp;

# Continue
Now you can continue with the Acute Lymphoblastic Leukemia oneAPI Classifier [Raspberry Pi 4 usage guide](../usage/raspberry-pi.md).

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and youlcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](../../CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss") President/Founder & Lead Developer, Sabadell, Spain

&nbsp;

# Versioning

You use SemVer for versioning. For the versions available, see [Releases](../../releases "Releases").

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](../../LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues

You use the [repo issues](../../issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](../../CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.