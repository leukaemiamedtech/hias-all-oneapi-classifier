#!/usr/bin/env python
""" Abstract class representing a HIAS AI OpenVINO Model.

Represents a HIAS AI OpenVINO Model. HIAS AI OpenVINO Models are used by AI Agents to process
incoming data.

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
- Adam Milton-Barker - First version - 2021-5-3

"""

import cv2
import os
import time

import numpy as np

from modules.AbstractOpenVINO import AbstractOpenVINO


class model_openvino(AbstractOpenVINO):
	""" Class representing a HIAS AI OpenVINO Model.

	This object represents a HIAS AI OpenVINO Model. HIAS AI OpenVINO Models are used by AI Agents
	to process incoming data.
	"""

	def load(self):
		""" Loads the model """

		mxml = self.helpers.confs["rpi4"]["ir"]
		mbin = os.path.splitext(mxml)[0] + ".bin"

		self.net = cv2.dnn.readNet(mxml, mbin)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

		self.helpers.logger.info("OpenVINO loaded.")

	def setBlob(self, frame):
		""" Gets a blob from the color frame """

		blob = cv2.dnn.blobFromImage(frame, self.helpers.confs["rpi4"]["inScaleFactor"],
									size=(self.imsize, self.imsize),
									mean=(self.helpers.confs["rpi4"]["meanVal"],
											self.helpers.confs["rpi4"]["meanVal"],
											self.helpers.confs["rpi4"]["meanVal"]),
									swapRB=True, crop=False)

		self.net.setInput(blob)

	def forwardPass(self):
		""" Gets a blob from the color frame """

		out = self.net.forward()

		return out

	def predict(self):
		""" Gets a prediction for an image. """

		predictions = self.forwardPass()
		predictions = predictions[0]
		idx = np.argsort(predictions)[::-1][0]
		prediction = self.helpers.confs["data"]["labels"][idx]

		return prediction

	def test(self):
		""" Test mode

		Loops through the test directory and classifies the images.
		"""

		files = 0
		tp = 0
		fp = 0
		tn = 0
		fn = 0
		totaltime = 0

		for testFile in os.listdir(self.testing_dir):
			if os.path.splitext(testFile)[1] in self.valid:

				files += 1
				fileName = self.testing_dir + "/" + testFile

				img = cv2.imread(fileName)
				self.helpers.logger.info("Loaded test image " + fileName)
				self.setBlob(self.resize(img))
				start = time.time()
				prediction = self.predict()
				end = time.time()
				benchmark = end - start
				totaltime += benchmark

				msg = ""
				if prediction == 1 and "_1." in testFile:
					tp += 1
					msg = "Acute Lymphoblastic Leukemia correctly detected (True Positive) in " + str(benchmark) + " seconds."
				elif prediction == 1 and "_0." in testFile:
					fp += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in " + str(benchmark) + " seconds."
				elif prediction == 0 and "_0." in testFile:
					tn += 1
					msg = "Acute Lymphoblastic Leukemia correctly not detected (True Negative) in " + str(benchmark) + " seconds."
				elif prediction == 0 and "_1." in testFile:
					fn += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly not detected (False Negative) in " + str(benchmark) + " seconds."
				self.helpers.logger.info(msg)

		self.helpers.logger.info("Images Classifier: " + str(files))
		self.helpers.logger.info("True Positives: " + str(tp))
		self.helpers.logger.info("False Positives: " + str(fp))
		self.helpers.logger.info("True Negatives: " + str(tn))
		self.helpers.logger.info("False Negatives: " + str(fn))
		self.helpers.logger.info("Total Time Taken: " + str(totaltime))

	def test_http(self):
		""" HTTP test mode

		Loops through the test directory and classifies the images
		by sending data to the classifier using HTTP requests.
		"""

		totaltime = 0
		files = 0

		tp = 0
		fp = 0
		tn = 0
		fn = 0

		self.addr = "http://" + self.helpers.credentials["server"]["ip"] + \
			':'+str(self.helpers.credentials["server"]["port"]) + '/Inference'
		self.headers = {'content-type': 'image/jpeg'}

		for testFile in os.listdir(self.testing_dir):
			if os.path.splitext(testFile)[1] in self.valid:

				start = time.time()
				prediction = self.http_request(self.testing_dir + "/" + testFile)
				end = time.time()
				benchmark = end - start
				totaltime += benchmark

				msg = ""
				status = ""
				outcome = ""

				if prediction["Diagnosis"] == "Positive" and "_1." in testFile:
					tp += 1
					status = "correctly"
					outcome = "(True Positive)"
				elif prediction["Diagnosis"] == "Positive" and "_0." in testFile:
					fp += 1
					status = "incorrectly"
					outcome = "(False Positive)"
				elif prediction["Diagnosis"] == "Negative" and "_0." in testFile:
					tn += 1
					status = "correctly"
					outcome = "(True Negative)"
				elif prediction["Diagnosis"] == "Negative" and "_1." in testFile:
					fn += 1
					status = "incorrectly"
					outcome = "(False Negative)"

				files += 1
				self.helpers.logger.info("Acute Lymphoblastic Leukemia " + status +
								" detected " + outcome + " in " + str(benchmark) + " seconds.")

		self.helpers.logger.info("Images Classified: " + str(files))
		self.helpers.logger.info("True Positives: " + str(tp))
		self.helpers.logger.info("False Positives: " + str(fp))
		self.helpers.logger.info("True Negatives: " + str(tn))
		self.helpers.logger.info("False Negatives: " + str(fn))
		self.helpers.logger.info("Total Time Taken: " + str(totaltime))

	def resize(self, img):
		""" Reshapes an image. """

		img = cv2.resize(img, (self.helpers.confs["data"]["dim"],
                         self.helpers.confs["data"]["dim"]))

		return img
