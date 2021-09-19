#!/usr/bin/env python
""" Abstract class representing a HIAS AI OpenVINO Model.

Represents a HIAS AI OpenVINO Model. HIAS AI OpenVINO
Models are used by AI Agents to process incoming data.

MIT License

Copyright (c) 2021 Asociación de Investigacion en Inteligencia Artificial
Para la Leucemia Peter Moss

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contributors:
- Adam Milton-Barker

"""

import cv2
import json
import os
import requests

from abc import ABC, abstractmethod


class AbstractOpenVINO(ABC):
    """ Abstract class representing a HIAS AI OpenVINO Model.

    This object represents a HIAS AI Model. HIAS AI Models
    are used by AI Agents to process incoming data.
    """

    def __init__(self, helpers):
        "Initializes the AbstractOpenVINO object."
        super().__init__()

        self.helpers = helpers
        self.confs = self.helpers.confs

        self.testing_dir = self.helpers.confs["data"]["test"]
        self.valid = self.helpers.confs["data"]["valid_types"]

        self.imsize = self.helpers.confs["data"]["dim"]

        self.helpers.logger.info(
            "OpenVINO class initialization complete.")

    def http_request(self, img_path):
        """ Sends image to the inference API endpoint. """

        self.helpers.logger.info(
            "Sending request for: " + img_path)

        _, img_encoded = cv2.imencode('.png', cv2.imread(img_path))
        response = requests.post(
            self.addr, data=img_encoded.tostring(), headers=self.headers)
        response = json.loads(response.text)

        return response

    @abstractmethod
    def load(self):
        """ Loads the model """
        pass

    @abstractmethod
    def setBlob(self, frame):
        """ Gets a blob from the color frame """
        pass

    @abstractmethod
    def forwardPass(self):
        """ Gets a blob from the color frame """
        pass

    @abstractmethod
    def predict(self):
        """ Gets a prediction for an image. """
        pass

    @abstractmethod
    def test(self):
        """Local test mode

        Loops through the test directory and classifies the images.
        """
        pass

    @abstractmethod
    def test_http(self):
        """ HTTP test mode

        Loops through the test directory and classifies the images
        by sending data to the classifier using HTTP requests.
        """
        pass

    @abstractmethod
    def resize(self, img):
        """ Resizes an image. """
        pass
