#!/usr/bin/env python
""" Server/API class.

Class for the HIAS IoT Agent server/API.

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
import jsonpickle

import numpy as np

from modules.AbstractServer import AbstractServer

from io import BytesIO
from PIL import Image
from flask import Flask, request, Response

class server(AbstractServer):
    """ Server/API class.

    Class for the HIAS IoT Agent server/API.
    """

    def predict(self, req):
        """ Classifies an image sent via HTTP. """

        img = np.fromstring(req.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (self.model.data.dim,
                               self.model.data.dim))
        img = self.model.reshape(img)

        return self.model.predict(img)

    def predict_openvino(self, req):
        """ Classifies an image sent via HTTP using OpenVINO. """

        if len(req.files) != 0:
            img = np.fromstring(req.files['file'].read(), np.uint8)
        else:
            img = np.fromstring(req.data, np.uint8)

        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

        img = self.model.resize(img)
        self.model.setBlob(img)

        return self.model.predict()

    def start(self):
        """ Starts the server. """

        app = Flask(self.helpers.credentials["iotJumpWay"]["name"])

        @app.route('/Inference', methods=['POST'])
        def Inference():
            """ Responds to HTTP POST requests. """

            self.mqtt.publish("States", {
                "Type": "Prediction",
                "Name": self.helpers.credentials["iotJumpWay"]["name"],
                "State": "Processing",
                "Message": "Processing data"
            })

            message = ""
            if self.model_type == "CNN":
                prediction = self.predict(request)
            else:
                prediction = self.predict_openvino(request)

            if prediction == 1:
                message = "Acute Lymphoblastic Leukemia detected!"
                diagnosis = "Positive"
            elif prediction == 0:
                message = "Acute Lymphoblastic Leukemia not detected!"
                diagnosis = "Negative"

            self.mqtt.publish("States", {
                "Type": "Prediction",
                "Name": self.helpers.credentials["iotJumpWay"]["name"],
                "State": diagnosis,
                "Message": message
            })

            resp = jsonpickle.encode({
                'Response': 'OK',
                'Message': message,
                'Diagnosis': diagnosis
            })

            return Response(
       response=resp, status=200, mimetype="application/json")

        app.run(
            host=self.helpers.get_ip_addr(),
            port=self.helpers.credentials["server"]["port"])