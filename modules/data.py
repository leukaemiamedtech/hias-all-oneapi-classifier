#!/usr/bin/env python
""" HIAS AI Model Data Class.

Provides the HIAS AI Model with the required required data
processing functionality.

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

import cv2
import os
import pathlib

import numpy as np

from numpy.random import seed
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from modules.AbstractData import AbstractData
from modules.augmentation import augmentation

class data(AbstractData):
	""" HIAS AI Model Data Class.

	Provides the HIAS AI Model with the required required data
	processing functionality.
	"""

	def pre_process(self):
		""" Processes the images. """

		aug = augmentation(self.helpers)

		data_dir = pathlib.Path(
			self.helpers.confs["data"]["train_dir"])
		data = list(data_dir.glob(
			'*' + self.helpers.confs["data"]["file_type"]))

		count = 0
		neg_count = 0
		pos_count = 0

		augmented_data = []
		augmented_labels = []

		for rimage in data:
			fpath = str(rimage)
			fname = os.path.basename(rimage)
			label = 0 if "_0" in fname else 1

			image = self.resize(fpath, self.dim)

			if image.shape[2] == 1:
				image = np.dstack(
					[image, image, image])

			augmented_data.append(image.astype(np.float32)/255.)
			augmented_labels.append(label)

			augmented_data.append(aug.grayscale(image))
			augmented_labels.append(label)

			augmented_data.append(aug.equalize_hist(image))
			augmented_labels.append(label)

			horizontal, vertical = aug.reflection(image)
			augmented_data.append(horizontal)
			augmented_labels.append(label)

			augmented_data.append(vertical)
			augmented_labels.append(label)

			augmented_data.append(aug.gaussian(image))
			augmented_labels.append(label)

			augmented_data.append(aug.translate(image))
			augmented_labels.append(label)

			augmented_data.append(aug.shear(image))
			augmented_labels.append(label)

			self.data, self.labels = aug.rotation(
				image, label, augmented_data, augmented_labels)

			if "_0" in fname:
				neg_count += 9
			else:
				pos_count += 9
			count += 9

		self.shuffle()
		self.convert_data()
		self.encode_labels()

		self.helpers.logger.info("Raw data: " + str(count))
		self.helpers.logger.info("Raw negative data: " + str(neg_count))
		self.helpers.logger.info("Raw positive data: " + str(pos_count))
		self.helpers.logger.info("Augmented data: " + str(self.data.shape))
		self.helpers.logger.info("Labels: " + str(self.labels.shape))

		self.get_split()

	def convert_data(self):
		""" Converts the training data to a numpy array. """

		self.data = np.array(self.data)
		self.helpers.logger.info("Data shape: " + str(self.data.shape))

	def encode_labels(self):
		""" One Hot Encodes the labels. """

		encoder = OneHotEncoder(categories='auto')

		self.labels = np.reshape(self.labels, (-1, 1))
		self.labels = encoder.fit_transform(self.labels).toarray()
		self.helpers.logger.info("Labels shape: " + str(self.labels.shape))

	def shuffle(self):
		""" Shuffles the data and labels. """

		self.data, self.labels = shuffle(
			self.data, self.labels, random_state=self.seed)

	def get_split(self):
		""" Splits the data and labels creating training and validation datasets. """

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			self.data, self.labels, test_size=0.255, random_state=self.seed)

		self.helpers.logger.info("Training data: " + str(self.X_train.shape))
		self.helpers.logger.info("Training labels: " + str(self.y_train.shape))
		self.helpers.logger.info("Validation data: " + str(self.X_test.shape))
		self.helpers.logger.info("Validation labels: " + str(self.y_test.shape))

	def resize(self, path, dim):
		""" Resizes an image to the provided dimensions (dim). """

		return cv2.resize(cv2.imread(path), (dim, dim))
