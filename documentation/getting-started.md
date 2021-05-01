# Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss
## Acute Lymphoblastic Leukemia oneAPI Classifier
### Getting Started

![Acute Lymphoblastic Leukemia oneAPI Classifier](../assets/images/all-oneapi-classifier-2020.png)

&nbsp;

# Table Of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
	- [Data Agumentation](#data-augmentation)
	- [Additional testing data](#additional-testing-data)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;

# Introduction
This guide will guide you through the installation process for the Acute Lymphoblastic Leukemia oneAPI Classifier.

&nbsp;

# Installation
Follow the [installation guide](Documentation/Installation.md) to install the requirements for this project.

&nbsp;

# Data
You need to be granted access to use the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. You can find the application form and information about getting access to the dataset on [this page](https://homes.di.unimi.it/scotti/all/#download) as well as information on how to contribute back to the project [here](https://homes.di.unimi.it/scotti/all/results.php). If you are not able to obtain a copy of the dataset please feel free to try this tutorial on your own dataset.

Once you have your data you need to add it to the project filesystem. You will notice the data folder in the Model directory, **model/data**, inside you have **train** & **test**.

## Data Augmentation

You will create an augmented dataset based on the [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon. In this case, you will use more rotated images to increase the dataset further.

## Additional testing data

Take the ten positive and ten negative samples shown below, and place them in the **model/data/test** directory. This will be used for testing the model, and also in our detection systems if you use this model with them.

To ensure you get the same results, please use the same test images, these same test images are used in our detection systems also. You can also try with your own image selction, however results may vary.

- im006_1.jpg
- im020_1.jpg
- im024_1.jpg
- im026_1.jpg
- im028_1.jpg
- im031_1.jpg
- im035_0.jpg
- im041_0.jpg
- im047_0.jpg
- im053_1.jpg
- im057_1.jpg
- im060_1.jpg
- im063_1.jpg
- im069_0.jpg
- im074_0.jpg
- im088_0.jpg
- im095_0.jpg
- im099_0.jpg
- im101_0.jpg
- im106_0.jpg

Next add the remaining 88 images to the **model/data/train** folder. The test images used will not be augmented and will not be used during model training.

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and youlcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss") President/Founder & Lead Developer, Sabadell, Spain

&nbsp;

# Versioning

You use SemVer for versioning. For the versions available, see [Releases](../releases "Releases").

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues

You use the [repo issues](../issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.