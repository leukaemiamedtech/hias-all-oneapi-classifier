# Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss
## Acute Lymphoblastic Leukemia oneAPI Classifier
### Getting Started

![Acute Lymphoblastic Leukemia oneAPI Classifier](../assets/images/all-oneapi-classifier-2020.png)

&nbsp;

# Table Of Contents

- [Introduction](#introduction)
    - [Network Architecture](#network-architecture)
- [Installation](#installation)
- [Data](#data)
	- [Data Augmentation](#data-augmentation)
	- [Application Testing Data](#application-testing-data)
- [Configuration](#configuration)
- [HIAS](#hias)
    - [AI Model](#ai-model)
    - [AI Agent](#ai-agent)
- [Training](#training)
    - [Metrics](#metrics)
	- [Start Training](#start-training)
	- [Training Data](#training-data)
	- [Model Summary](#model-summary)
	- [Training Results](#training-results)
	- [Metrics Overview](#metrics-overview)
	- [ALL-IDB Required Metrics](#all-idb-required-metrics)
- [Testing](#testing)
- [Convert Model](#convert-model)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;

# Introduction
This guide will guide you through the installation process for the Acute Lymphoblastic Leukemia oneAPI Classifier.

## Network Architecture
You will build a Convolutional Neural Network based on the proposed architecture in [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System"). The network will consist of the following 5 layers (missing out the zero padding layers). Note you are usng an conv sizes of (100x100x30) whereas in the paper, the authors use (50x50x30).

- Conv layer (100x100x30)
- Conv layer (100x100x30)
- Max-Pooling layer (50x50x30)
- Fully Connected layer (2 neurons)
- Softmax layer (Output 2)

&nbsp;

# Installation
First you need to install the required software for training the model. Below are the available installation guides:

- [Ubuntu installation guide](installation/ubuntu.md) (Training).
- [Raspberry Pi 4 installation guide](installation/rpi4.md) (Inference on the edge).

&nbsp;

# Data
You need to be granted access to use the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. You can find the application form and information about getting access to the dataset on [this page](https://homes.di.unimi.it/scotti/all/#download) as well as information on how to contribute back to the project [here](https://homes.di.unimi.it/scotti/all/results.php).

_If you are not able to obtain a copy of the dataset please feel free to try this tutorial on your own dataset._

Once you have your data you need to add it to the project filesystem. You will notice the data folder in the Model directory, **model/data**, inside you have **train** & **test**. Add all of the images from the ALL_IDB1 dataset to the **model/data/train** folder.

## Data Augmentation

You will create an augmented dataset based on the [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon. In this case, you will use more rotated images to increase the dataset further.

## Application testing data

In the data processing stage, ten negative images and ten positive images are removed from the dataset and moved to the **model/data/test/** directory. This data is not seen by the network during the training process, and is used by applications.

To ensure your model gets the same results, please use the same test images. By default HIAS compatible projects will be expecting the same test images.  You can also try with your own image selection, however results may vary and you will need to make additional changes to our HIAS compatible projects.

To specify which test images to use modify the [configuration/config.json](../configuration/config.json) file as shown below:

```
"test_data": [
	"im006_1.jpg",
	"im020_1.jpg",
	"im024_1.jpg",
	"im026_1.jpg",
	"im028_1.jpg",
	"im031_1.jpg",
	"im035_0.jpg",
	"im041_0.jpg",
	"im047_0.jpg",
	"im053_1.jpg",
	"im057_1.jpg",
	"im060_1.jpg",
	"im063_1.jpg",
	"im069_0.jpg",
	"im074_0.jpg",
	"im088_0.jpg",
	"im095_0.jpg",
	"im099_0.jpg",
	"im101_0.jpg",
	"im106_0.jpg"
],
```

&nbsp;

# Configuration
[configuration/config.json](../configuration/config.json "configuration/config.json")  holds the configuration for our application.

- Change **agent->cores** to the number of cores you have on your training computer.
- Change **agent->server** to the local IP of your training device.
- Change **agent->port** to a different number.

```
{
    "agent": {
        "cores": 8,
        "server": "",
        "port": 1234,
        "params": [
            "train",
            "classify",
            "server",
            "classify_http"
        ]
    },
    "data": {
        "dim": 100,
        "file_type": ".jpg",
        "labels": [0, 1],
        "rotations": 10,
        "seed": 2,
        "split": 0.3,
        "test": "model/data/test",
        "test_data": [
            "Im006_1.jpg",
            "Im020_1.jpg",
            "Im024_1.jpg",
            "Im026_1.jpg",
            "Im028_1.jpg",
            "Im031_1.jpg",
            "Im035_0.jpg",
            "Im041_0.jpg",
            "Im047_0.jpg",
            "Im053_1.jpg",
            "Im057_1.jpg",
            "Im060_1.jpg",
            "Im063_1.jpg",
            "Im069_0.jpg",
            "Im074_0.jpg",
            "Im088_0.jpg",
            "Im095_0.jpg",
            "Im099_0.jpg",
            "Im101_0.jpg",
            "Im106_0.jpg"
        ],
        "train_dir": "model/data/train",
        "valid_types": [
            ".JPG",
            ".JPEG",
            ".PNG",
            ".GIF",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif"
        ]
    },
    "model": {
        "device": "CPU",
        "freezing_log_dir": "model/freezing",
        "frozen": "frozen.pb",
        "ir": "model/ir/frozen.xml",
        "model": "model/model.json",
        "saved_model_dir": "model",
        "weights": "model/weights.h5"
    },
    "train": {
        "batch": 100,
        "decay_adam": 1e-6,
        "epochs": 150,
        "learning_rate_adam": 1e-4,
        "val_steps": 10
    }
}
```

The configuration object contains 4 Json Objects (agent, data, model and train). Agent has the information used to set up your server, data has the configuration related to preparing the training and validation data, model holds the model file paths, and train holds the training parameters.

&nbsp;

# HIAS

This device is a HIAS AI Agent and uses the HIAS MQTT Broker to communicate with the HIAS network. To setup an AI Agent on the HIAS network, head to the HIAS UI.

The HIAS network is powered by a context broker that stores contextual data and exposes the data securely to authenticated HIAS applications and devices.

Each HIAS AI Agent & AI Model has a JSON representation stored in the HIAS Context Broker that holds their contextual information.

## AI Model

A HIAS AI Model is a JSON representation of an Artificial Intelligence model used by the HIAS network.

First you need to set a HIAS AI Model up in the HIAS UI. Navigate to **AI->Models->Create** to create a HIAS AI Model. A future release of HIAS will provide the functionality to import the HIAS JSON representation of the AI Model, but for now you have to manually create the AI Model in the UI.

![HIAS AI Model](../assets/images/hias-ai-model.jpg)

Once you have completed the form and submitted it, you can find the newly created AI Model by navigating to **AI->Models->List** and clicking on the relevant Model.

On the HIAS AI Model page you will be able to update the contextual data for the model, and also find the JSON representation.

![HIAS AI Model](../assets/images/hias-ai-model-edit.jpg)

## AI Agent

A HIAS AI Agent is a bridge between HIAS devices and applications, and HIAS IoT Agents. The AI Agents process incoming data by passing it through HIAS AI Models and returning the response back to the requesting device/application.

As with AI Models, AI Agents have an entry in the HIAS Context Broker and a JSON representation stored on the network.

You will now need to create your HIAS AI Agent and retrieve the credentials required by your Acute Lymphoblastic Leukemia oneAPI Classifier. Navigate to **AI->Agents->Create** to create a HIAS AI Model.

![HIAS AI Agent](../assets/images/hias-ai-agent.jpg)

**MAKE SURE YOU SELECT THE PREVIOUSLY CREATED HIAS AI MODEL**

Once you have completed the form and submitted it, you can find the newly created AI Agent by navigating to **AI->Agents->List** and clicking on the relevant Agent.

On the HIAS AI Agent page you will be able to update the contextual data for the agent, and also find the JSON representation.

![HIAS AI Model](../assets/images/hias-ai-agent-edit.jpg)

You now need to download the credentials required to connect the Acute Lymphoblastic Leukemia oneAPI Classifier to the HIAS network.

Click on the **Agent Credentials** section to download the credentials file. This should open your file browser, navigate to the **Acute-Lymphoblastic-Leukemia-oneAPI-Classifier/configuration/** directory and save the file as **credentials.json**.

&nbsp;

# Training
Now you are ready to train your model.

## Metrics
We can use metrics to measure the effectiveness of our model. In this network you will use the following metrics:

```
tf.keras.metrics.BinaryAccuracy(name='accuracy'),
tf.keras.metrics.Precision(name='precision'),
tf.keras.metrics.Recall(name='recall'),
tf.keras.metrics.AUC(name='auc')
```

These metrics will be displayed and plotted once our model is trained.  A useful tutorial while working on the metrics was the [Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) tutorial on Tensorflow's website.

## Start Training
Ensuring you have completed all previous steps, you can start training using the following command.

```
python agent.py train
```

This tells the application to start training the model.

## Training Data
First the training and validation data will be prepared.

```
2021-05-02 18: 43: 57, 164 - Agent - INFO - Data shape: (1584, 100, 100, 3)
2021-05-02 18: 43: 57, 165 - Agent - INFO - Labels shape: (1584, 2)
2021-05-02 18: 43: 57, 165 - Agent - INFO - Raw data: 792
2021-05-02 18: 43: 57, 165 - Agent - INFO - Raw negative data: 441
2021-05-02 18: 43: 57, 166 - Agent - INFO - Raw positive data: 351
2021-05-02 18: 43: 57, 166 - Agent - INFO - Augmented data: (1584, 100, 100, 3)
2021-05-02 18: 43: 57, 166 - Agent - INFO - Labels: (1584, 2)
2021-05-02 18: 43: 57, 334 - Agent - INFO - Training data: (1180, 100, 100, 3)
2021-05-02 18: 43: 57, 334 - Agent - INFO - Training labels: (1180, 2)
2021-05-02 18: 43: 57, 334 - Agent - INFO - Validation data: (404, 100, 100, 3)
2021-05-02 18: 43: 57, 334 - Agent - INFO - Validation labels: (404, 2)
2021-05-02 18: 43: 57, 359 - Agent - INFO - Data preperation complete.
```

### Model Summary

Before the model begins training, you will be shown the model summary.

```
Model: "AllOneApiClassifier"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
zero_padding2d (ZeroPadding2 (None, 104, 104, 3)       0
_________________________________________________________________
conv2d (Conv2D)              (None, 100, 100, 30)      2280
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 104, 104, 30)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 100, 100, 30)      22530
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 50, 50, 30)        0
_________________________________________________________________
flatten (Flatten)            (None, 75000)             0
_________________________________________________________________
dense (Dense)                (None, 2)                 150002
_________________________________________________________________
activation (Activation)      (None, 2)                 0
=================================================================
Total params: 174,812
Trainable params: 174,812
Non-trainable params: 0
_________________________________________________________________
2021-05-02 18:43:57,414 - Agent - INFO - Network initialization complete.
2021-05-02 18:43:57,414 - Agent - INFO - Using Adam Optimizer.
Train on 1180 samples, validate on 404 samples
```

Our network matches the architecture proposed in the paper.

## Training Results
Below are the training results for 150 epochs.

<img src="../model/plots/accuracy.png" alt="Adam Optimizer Results" />

_Fig 2. Accuracy_

<img src="../model/plots/loss.png" alt="Loss" />

_Fig 3. Loss_

<img src="../model/plots/precision.png" alt="Precision" />

_Fig 4. Precision_

<img src="../model/plots/recall.png" alt="Recall" />

_Fig 5. Recall_

<img src="../model/plots/auc.png" alt="AUC" />

_Fig 6. AUC_

<img src="../model/plots/confusion-matrix.png" alt="AUC" />

_Fig 7. Confusion Matrix_

```
2021-05-02 14:33:48,514 - Agent - INFO - Metrics: loss 0.04992164568145677
2021-05-02 14:33:48,514 - Agent - INFO - Metrics: acc 0.9826733
2021-05-02 14:33:48,514 - Agent - INFO - Metrics: precision 0.9826733
2021-05-02 14:33:48,515 - Agent - INFO - Metrics: recall 0.9826733
2021-05-02 14:33:48,515 - Agent - INFO - Metrics: auc 0.99847436

2021-05-02 14:33:49,146 - Agent - INFO - Confusion Matrix: [[220   1] [  6 177]]

2021-05-02 14:33:49,278 - Agent - INFO - True Positives: 177(43.81188118811881%)
2021-05-02 14:33:49,278 - Agent - INFO - False Positives: 1(0.24752475247524752%)
2021-05-02 14:33:49,278 - Agent - INFO - True Negatives: 220(54.45544554455446%)
2021-05-02 14:33:49,278 - Agent - INFO - False Negatives: 6(1.4851485148514851%)
2021-05-02 14:33:49,278 - Agent - INFO - Specificity: 0.995475113122172
2021-05-02 14:33:49,278 - Agent - INFO - Misclassification: 7(1.7326732673267327%)
```

## Metrics Overview
| Accuracy | Recall | Precision | AUC/ROC |
| ---------- | ---------- | ---------- | ---------- |
| 0.9826733 | 0.9826733 | 0.9826733 | 0.99847436 |


## ALL-IDB Required Metrics
| Figures of merit     | Amount/Value | Percentage |
| -------------------- | ----- | ---------- |
| True Positives       | 177 | 43.81188118811881% |
| False Positives      | 1 | 0.24752475247524752% |
| True Negatives       | 220 | 54.45544554455446% |
| False Negatives      | 6 | 1.4851485148514851% |
| Misclassification    | 7 | 1.7326732673267327% |
| Sensitivity / Recall | 0.9826733   | 0.98% |
| Specificity          | 0.995475113122172  | 100% |

&nbsp;

# Testing

Now you will test the classifier on your training machine. You will use the 20 images that were removed from the training data in a previous part of this tutorial.

To run the AI Agent in test mode use the following command:

```
python3 agenty.py classify
```

You should see the following which shows you the network architecture:

```
Model: "AllOneApiClassifier"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
zero_padding2d (ZeroPadding2 (None, 104, 104, 3)       0
_________________________________________________________________
conv2d (Conv2D)              (None, 100, 100, 30)      2280
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 104, 104, 30)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 100, 100, 30)      22530
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 50, 50, 30)        0
_________________________________________________________________
flatten (Flatten)            (None, 75000)             0
_________________________________________________________________
dense (Dense)                (None, 2)                 150002
_________________________________________________________________
activation (Activation)      (None, 2)                 0
=================================================================
Total params: 174,812
Trainable params: 174,812
Non-trainable params: 0
```

Finally the application will start processing the test images and the results will be displayed in the console.

```
2021-05-02 21:38:46,437 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.15969634056091309 seconds.
2021-05-02 21:38:46,472 - Agent - INFO - Loaded test image model/data/test/Im028_1.jpg
2021-05-02 21:38:46,494 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.05695939064025879 seconds.
2021-05-02 21:38:46,577 - Agent - INFO - Loaded test image model/data/test/Im106_0.jpg
2021-05-02 21:38:46,598 - Agent - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.10403966903686523 seconds.
2021-05-02 21:38:46,680 - Agent - INFO - Loaded test image model/data/test/Im101_0.jpg
2021-05-02 21:38:46,702 - Agent - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.10358524322509766 seconds.
2021-05-02 21:38:46,736 - Agent - INFO - Loaded test image model/data/test/Im024_1.jpg
2021-05-02 21:38:46,757 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.05475354194641113 seconds.
2021-05-02 21:38:46,839 - Agent - INFO - Loaded test image model/data/test/Im074_0.jpg
2021-05-02 21:38:46,864 - Agent - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.1065070629119873 seconds.
2021-05-02 21:38:46,946 - Agent - INFO - Loaded test image model/data/test/Im035_0.jpg
2021-05-02 21:38:46,970 - Agent - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.1056666374206543 seconds.
2021-05-02 21:38:47,003 - Agent - INFO - Loaded test image model/data/test/Im006_1.jpg
2021-05-02 21:38:47,024 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.05464029312133789 seconds.
2021-05-02 21:38:47,058 - Agent - INFO - Loaded test image model/data/test/Im020_1.jpg
2021-05-02 21:38:47,079 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.05452728271484375 seconds.
2021-05-02 21:38:47,151 - Agent - INFO - Loaded test image model/data/test/Im095_0.jpg
2021-05-02 21:38:47,173 - Agent - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in 0.09325885772705078 seconds.
2021-05-02 21:38:47,255 - Agent - INFO - Loaded test image model/data/test/Im069_0.jpg
2021-05-02 21:38:47,276 - Agent - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.1036827564239502 seconds.
2021-05-02 21:38:47,310 - Agent - INFO - Loaded test image model/data/test/Im031_1.jpg
2021-05-02 21:38:47,331 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.05464768409729004 seconds.
2021-05-02 21:38:47,414 - Agent - INFO - Loaded test image model/data/test/Im099_0.jpg
2021-05-02 21:38:47,436 - Agent - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.10416984558105469 seconds.
2021-05-02 21:38:47,469 - Agent - INFO - Loaded test image model/data/test/Im026_1.jpg
2021-05-02 21:38:47,491 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.05491828918457031 seconds.
2021-05-02 21:38:47,573 - Agent - INFO - Loaded test image model/data/test/Im057_1.jpg
2021-05-02 21:38:47,598 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.10681462287902832 seconds.
2021-05-02 21:38:47,681 - Agent - INFO - Loaded test image model/data/test/Im088_0.jpg
2021-05-02 21:38:47,703 - Agent - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.1050255298614502 seconds.
2021-05-02 21:38:47,785 - Agent - INFO - Loaded test image model/data/test/Im060_1.jpg
2021-05-02 21:38:47,807 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.10350656509399414 seconds.
2021-05-02 21:38:47,890 - Agent - INFO - Loaded test image model/data/test/Im053_1.jpg
2021-05-02 21:38:47,912 - Agent - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative) in 0.10537457466125488 seconds.
2021-05-02 21:38:47,995 - Agent - INFO - Loaded test image model/data/test/Im041_0.jpg
2021-05-02 21:38:48,016 - Agent - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.10396504402160645 seconds.
2021-05-02 21:38:48,099 - Agent - INFO - Loaded test image model/data/test/Im047_0.jpg
2021-05-02 21:38:48,120 - Agent - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.10404443740844727 seconds.
2021-05-02 21:38:48,121 - Agent - INFO - Images Classified: 20
2021-05-02 21:38:48,121 - Agent - INFO - True Positives: 9
2021-05-02 21:38:48,121 - Agent - INFO - False Positives: 1
2021-05-02 21:38:48,121 - Agent - INFO - True Negatives: 9
2021-05-02 21:38:48,121 - Agent - INFO - False Negatives: 1
2021-05-02 21:38:48,121 - Agent - INFO - Total Time Taken: 1.8397836685180664
```

In the current terminal, now use the following command:

```
python3 agenty.py server
```

This will start the server on your training machine that exposes the model via a REST API. Now open a new terminal, navigate to the project root and use the following command:

```
python3 agenty.py classify_http
```

This will start agent in HTTP Inference mode. The agent will loop through the testing data and send each image to the server for classification, the results are then displayed in the console.

```
2021-05-02 21:39:18,965 - Agent - INFO - Sending request for: model/data/test/Im063_1.jpg
2021-05-02 21:39:19,462 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.4971804618835449 seconds.
2021-05-02 21:39:19,462 - Agent - INFO - Sending request for: model/data/test/Im028_1.jpg
2021-05-02 21:39:19,691 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.22867822647094727 seconds.
2021-05-02 21:39:19,691 - Agent - INFO - Sending request for: model/data/test/Im106_0.jpg
2021-05-02 21:39:20,104 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Negative) in 0.41293954849243164 seconds.
2021-05-02 21:39:20,104 - Agent - INFO - Sending request for: model/data/test/Im101_0.jpg
2021-05-02 21:39:20,517 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Negative) in 0.41324901580810547 seconds.
2021-05-02 21:39:20,517 - Agent - INFO - Sending request for: model/data/test/Im024_1.jpg
2021-05-02 21:39:20,716 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.19890832901000977 seconds.
2021-05-02 21:39:20,716 - Agent - INFO - Sending request for: model/data/test/Im074_0.jpg
2021-05-02 21:39:21,128 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Negative) in 0.41187310218811035 seconds.
2021-05-02 21:39:21,128 - Agent - INFO - Sending request for: model/data/test/Im035_0.jpg
2021-05-02 21:39:21,539 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Negative) in 0.4105658531188965 seconds.
2021-05-02 21:39:21,539 - Agent - INFO - Sending request for: model/data/test/Im006_1.jpg
2021-05-02 21:39:21,740 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.20087504386901855 seconds.
2021-05-02 21:39:21,740 - Agent - INFO - Sending request for: model/data/test/Im020_1.jpg
2021-05-02 21:39:21,946 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.20533061027526855 seconds.
2021-05-02 21:39:21,946 - Agent - INFO - Sending request for: model/data/test/Im095_0.jpg
2021-05-02 21:39:22,325 - Agent - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in 0.3794980049133301 seconds.
2021-05-02 21:39:22,325 - Agent - INFO - Sending request for: model/data/test/Im069_0.jpg
2021-05-02 21:39:22,740 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Negative) in 0.4143247604370117 seconds.
2021-05-02 21:39:22,740 - Agent - INFO - Sending request for: model/data/test/Im031_1.jpg
2021-05-02 21:39:22,939 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.19963884353637695 seconds.
2021-05-02 21:39:22,939 - Agent - INFO - Sending request for: model/data/test/Im099_0.jpg
2021-05-02 21:39:23,350 - Agent - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in 0.41005873680114746 seconds.
2021-05-02 21:39:23,350 - Agent - INFO - Sending request for: model/data/test/Im026_1.jpg
2021-05-02 21:39:23,550 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.20075702667236328 seconds.
2021-05-02 21:39:23,551 - Agent - INFO - Sending request for: model/data/test/Im057_1.jpg
2021-05-02 21:39:23,960 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.4094827175140381 seconds.
2021-05-02 21:39:23,960 - Agent - INFO - Sending request for: model/data/test/Im088_0.jpg
2021-05-02 21:39:24,372 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Negative) in 0.4121434688568115 seconds.
2021-05-02 21:39:24,372 - Agent - INFO - Sending request for: model/data/test/Im060_1.jpg
2021-05-02 21:39:24,784 - Agent - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Negative) in 0.41173434257507324 seconds.
2021-05-02 21:39:24,784 - Agent - INFO - Sending request for: model/data/test/Im053_1.jpg
2021-05-02 21:39:25,206 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.4215230941772461 seconds.
2021-05-02 21:39:25,206 - Agent - INFO - Sending request for: model/data/test/Im041_0.jpg
2021-05-02 21:39:25,617 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Negative) in 0.41069746017456055 seconds.
2021-05-02 21:39:25,617 - Agent - INFO - Sending request for: model/data/test/Im047_0.jpg
2021-05-02 21:39:26,032 - Agent - INFO - Acute Lymphoblastic Leukemia correctly detected (True Negative) in 0.4154655933380127 seconds.
2021-05-02 21:39:26,032 - Agent - INFO - Images Classified: 20
2021-05-02 21:39:26,032 - Agent - INFO - True Positives: 9
2021-05-02 21:39:26,033 - Agent - INFO - False Positives: 2
2021-05-02 21:39:26,033 - Agent - INFO - True Negatives: 8
2021-05-02 21:39:26,033 - Agent - INFO - False Negatives: 1
2021-05-02 21:39:26,033 - Agent - INFO - Total Time Taken: 7.064924240112305
```

# Convert Model

Now you need to convert your frozen model to an Intermediate Representation. To do this, use the following command, replacing **YourProjectPath** with the path to your project home.

```
python3 /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py --input_model /YourProjectPath/ALL-Classifier-2020/model/freezing/frozen.pb --input_shape [1,100,100,3] --output_dir /YourProjectPath/ALL-Classifier-2020/model/ir --reverse_input_channels --generate_deprecated_IR_V7
```

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