#!/usr/bin/env python
""" Abstract class representing a HIAS AI Model.

Represents a HIAS AI Model. HIAS AI Models are used by AI Agents to process
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
- Adam Milton-Barker - First version - 2021-5-1

"""

import os
import time

from abc import ABC, abstractmethod

from modules.data import data

class AbstractModel(ABC):
	""" Abstract class representing a HIAS AI Model.

	This object represents a HIAS AI Model. HIAS AI Models are used by AI Agents
	to process incoming data.
	"""

	def __init__(self, helpers):
		"Initializes the AbstractModel object."
		super().__init__()

		self.helpers = helpers
		self.confs = self.helpers.confs

		self.data = data(self.helpers)

		self.helpers.logger.info("Model class initialization complete.")

	@abstractmethod
	def prepare(self):
		""" Prepares the model

		Loops through the test directory and classifies the images.
		"""
		pass

	@abstractmethod
	def train(self):
		""" Trains the model

		Trains the neural network.
		"""
		pass

	@abstractmethod
	def load(self):
		""" Loads the model

		Loops through the test directory and classifies the images.
		"""
		pass

	@abstractmethod
	def predict(self, img):
		""" Gets a prediction for an image. """
		pass

	@abstractmethod
	def test(self, img):
		"""Local test mode

		Loops through the test directory and classifies the images.
		"""
		pass
