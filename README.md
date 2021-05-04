# Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss
## Acute Lymphoblastic Leukemia oneAPI Classifier

![Acute Lymphoblastic Leukemia oneAPI Classifier](assets/images/all-oneapi-classifier-2020.png)

[![CURRENT RELEASE](https://img.shields.io/badge/CURRENT%20RELEASE-1.0.0-blue.svg)](https://github.com/AIIAL/Acute-Lymphoblastic-Leukemia-oneAPI-Classifier/tree/1.0.0) [![UPCOMING RELEASE](https://img.shields.io/badge/CURRENT%20DEV%20BRANCH-1.1.0-blue.svg)](https://github.com/AIIAL/Acute-Lymphoblastic-Leukemia-oneAPI-Classifier/tree/1.1.0) [![Contributions Welcome!](https://img.shields.io/badge/Contributions-Welcome-lightgrey.svg)](CONTRIBUTING.md)  [![Issues](https://img.shields.io/badge/Issues-Welcome-lightgrey.svg)](issues) [![LICENSE](https://img.shields.io/badge/LICENSE-MIT-blue.svg)](LICENSE)

&nbsp;

# Table Of Contents

- [Introduction](#introduction)
- [DISCLAIMER](#disclaimer)
- [Motivation](#motivation)
- [Acute Lymphoblastic Leukemia](#acute-lymphoblastic-leukemia)
  - [ALL-IDB](#all-idb)
    - [ALL_IDB1](#all_idb1)
- [Acute Lymphoblastic Leukemia Tensorflow Classifier 2020](#acute-lymphoblastic-leukemia-tensorflow-classifier-2020)
- [Intel Technologies](#intel-technologies)
  - [Intel® oneAPI Toolkits (Beta)](#intel-oneapi-toolkits-beta)
  - [Intel® Distribution for Python](#intel-distribution-for-python)
  - [Intel® Optimization for TensorFlow](#intel-optimization-for-tensorflow)
  - [Intel® Distribution of OpenVINO™ Toolkit](#intel-distribution-of-openvino-toolkit)
  - [Intel® Movidius™ Neural Compute Stick 2](#intel-movidius-neural-compute-stick-2)
- [Acute Lymphoblastic Leukemia oneAPI Classifier 2021](#acute-lymphoblastic-leukemia-oneapi-classifier-2021)
- [GETTING STARTED](#getting-started)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;

# Introduction

The **Acute Lymphoblastic Leukemia (ALL) oneAPI Classifier** is an open-source classifier programmed using the  [Intel® Distribution for Python*](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html) and trained using [Intel® Optimization for TensorFlow*](https://software.intel.com/content/www/us/en/develop/articles/intel-optimization-for-tensorflow-installation-guide.html) (Tensorflow 2.1). The model is deployed on a Raspberry 4 using [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) and inference is carried out using the Intel Movidius Neural Compute Stick 2.

&nbsp;

# DISCLAIMER

This project should be used for research purposes only. The purpose of the project is to show the potential of Artificial Intelligence for medical support systems such as diagnosis systems.

Although the classifier is accurate and shows good results both on paper and in real world testing, it is not meant to be an alternative to professional medical diagnosis.

Developers that have contributed to this repository have experience in using Artificial Intelligence for detecting certain types of cancer. They are not doctors, medical or cancer experts.

&nbsp;

# Motivation

The motivation for this project came from the interest in exploring how Intel technologies could be used to create an improved version of the [Acute Lymphoblastic Leukemia Tensorflow 2020](#all-tensorflow-2020) project.

The goal is to create an improved computer vision model that was capable of detecting Acute Lymphoblastic Leukemia in unseen images of periphial blood samples with high accuracy and efficiency, running a low powered/low resources Raspberry Pi 4.

&nbsp;

# Acute Lymphoblastic Leukemia
Acute Lymphoblastic Leukemia (ALL), also known as Acute Lymphocytic Leukemia, is a cancer that affects the Lymphoid blood cell lineage. Unlike AML, ALL only affects the white blood cells, namely, Lymphocytes. Lymphocytes include B Cells, T Cells and NK (Natural Killer) cells. ALL is caused by Lymphoid Blasts, or Lymphoblasts, developing into immature Lymphocytes, and an abnormal amount of these immature Lymphocytes are produced. Lymphocytes are white blood cells and play a very important role in the immune system helping to fight off diseases.

Acute Lymphoblastic Leukemia is most commonly found in children, and is the most common form of child cancer, with around 3000 cases a year in the US alone. Like Acute Myeloid Leukemia, although common, it is still quite rare. In both children and adults, early detection is critical. Treatment must start imassetstely due to the aggressiveness of the cancer. [More info](https://www.leukemiaresearchassociation.ai/leukemia).

## ALL-IDB
You need to be granted access to use the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. You can find the application form and information about getting access to the dataset on [this page](https://homes.di.unimi.it/scotti/all/#download) as well as information on how to contribute back to the project [here](https://homes.di.unimi.it/scotti/all/results.php). If you are not able to obtain a copy of the dataset please feel free to try this tutorial on your own dataset, we would be very happy to find additional AML & ALL datasets.

### ALL_IDB1
In this project, [ALL-IDB1](https://homes.di.unimi.it/scotti/all/#datasets) is used, one of the datsets from the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. We will use data augmentation to increase the amount of training and testing data we have.

"The ALL_IDB1 version 1.0 can be used both for testing segmentation capability of algorithms, as well as the classification systems and image preprocessing methods. This dataset is composed of 108 images collected during September, 2005. It contains about 39000 blood elements, where the lymphocytes has been labeled by expert oncologists. The images are taken with different magnifications of the microscope ranging from 300 to 500."

&nbsp;

# Acute Lymphoblastic Leukemia Tensorflow Classifier 2020

The Acute Lymphoblastic Leukemia Tensorflow Classifier 2020 network architecture is based on the proposed architecture in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper. The model was trained and tested on a selection of different CPUs/GPUs, with the **Intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8** and **Windows 10** providing the most optimal results.

| OS | Hardware | Training | Validation | Test | Accuracy | Recall | Precision | AUC/ROC |
| -------------------- | -------------------- | -------------------- | ----- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Google Colab | Tesla K80 GPU | 1180 |  404 | 20 |  0.9727723 | 0.9727723 | 0.9727723 | 0.9948964 |
| Windows 10 | NVIDIA GeoForce GTX 1060 | 1180 |  404 | 20 |  0.97066015 | 0.97066015 | 0.97066015 | 0.9908836 |
| Ubuntu 18.04 | NVIDIA GTX 1050 Ti Ti/PCIe/SSE2 | 1180 |  404 | 20 |  0.97772276 | 0.97772276 | 0.97772276 | 0.9989155 |
| Ubuntu 18.04 | Intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8   | 1180 |  404 | 20 |  0.9752475 | 0.9752475 | 0.9752475 | 0.991492 |
| Windows 10 | Intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8   | 1180 |  404 | 20 |  0.9851485 | 0.9851485 | 0.9851485 | 0.9985846 |
| macOS Mojave 10.14.6 | Intel® Core™ i5 CPU @ 2.4 GHz   | 1180 |  404 | 20 |  0.9589041 | 0.9589041 | 0.9589041 | 0.99483955 |

Source: [Acute Lymphoblastic Leukemia Tensorflow Classifier 2020](https://github.com/AMLResearchProject/ALL-Tensorflow-2020#classifier)

&nbsp;

# Intel Technologies

## Intel® oneAPI Toolkits (Beta)
[Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html) are a collection of toolkits that provide the tools to optimize training and running inference on Artificial Intelligence models, maximizing the use of Intel architecture, including CPU, GPU, VPU and FPGA.

## Intel® Distribution for Python
[Intel® Distribution for Python](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html) enhances standard Python and helps to speed up popular AI packages such as Numpy, SciPy and Scikit-Learn.

## Intel® Optimization for TensorFlow
[Intel® Optimization for TensorFlow](https://software.intel.com/content/www/us/en/develop/articles/intel-optimization-for-tensorflow-installation-guide.html) optimizes the popular Tensorflow framework using Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN). Intel® MKL-DNN is an open-source library for enhancing performance by accelerating deep learning libraries such as Tensorflow on Intel architecture.

## Intel® Distribution of OpenVINO™ Toolkit
[Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) is based on Convolutional Neural Networks and optimizes models used on Intel CPUs/GPUs, VPUs, FPGA etc. Models are converted to [Intermediate Representations (IR)](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html) which allow them to be used with the [Inference Engine](https://docs.openvinotoolkit.org/2020.2/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html).

## Intel® Movidius™ Neural Compute Stick 2
The [Intel® Movidius™ Neural Compute Stick 2](https://software.intel.com/content/www/us/en/develop/hardware/neural-compute-stick.html) is a USB plug & play AI device for deep learning inference at the edge. Combined with the Intel® OpenVINO™ Toolkit, developers can develop, fine-tune, and deploy convolutional neural networks (CNNs) on low-power applications that require real-time inference.

&nbsp;

# Acute Lymphoblastic Leukemia oneAPI Classifier 2021

To create the newly improved **Acute Lymphoblastic Leukemia oneAPI Classifier 2021** we will take the following steps:

- Use the data augmentation techniques proposed in [the Leukemia Blood Cell Image Classification Using Convolutional Neural Network by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon](http://www.ijcte.org/vol10/1198-H0012.pdf).

- Install [Intel® Optimization for TensorFlow*](https://software.intel.com/content/www/us/en/develop/articles/intel-optimization-for-tensorflow-installation-guide.html) on our training device. In the case of this tutorial, a Linux machine with an NVIDIA GTX 1050 Ti was used for training.

- Install [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) on our inference device.  In the case of this tutorial, the model is deployed to a Raspberry Pi 4 for inference.

- Train a model based on the architecture proposed in [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") using ALL_IDB1 from the [Acute Lymphoblastic Leukemia Image Database for Image Processing dataset](https://homes.di.unimi.it/scotti/all/#download).

- Test the model using commandline and classify unseen data using HTTP requests to a local API endpoint and via the HIAS network.

&nbsp;

# GETTING STARTED

Ready to get started ? Head over to the [Getting Started guide](documentation/getting-started.md) for instructions on how to download/install and setup the Acute Lymphoblastic Leukemia oneAPI Classifier 2021.

&nbsp;

# Contributing
The Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss encourages and welcomes code contributions, bug fixes and enhancements from the Github community.

Please read the [CONTRIBUTING](CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors
- [Adam Milton-Barker](https://www.leukemiaairesearch.com/association/volunteers/adam-milton-barker "Adam Milton-Barker") - [Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss") President/Founder & Lead Developer, Sabadell, Spain

&nbsp;

# Versioning
We use SemVer for versioning.

&nbsp;

# License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues
We use the [repo issues](issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.