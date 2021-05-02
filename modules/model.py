#!/usr/bin/env python
""" Class representing a HIAS AI Model.

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

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from modules.AbstractModel import AbstractModel


class model(AbstractModel):
	""" Class representing a HIAS AI Model.

	This object represents a HIAS AI Model.HIAS AI Models are used by AI Agents
	to process incoming data.
	"""

	def prepare_data(self):
		""" Creates/sorts dataset. """

		self.data.pre_process()

		self.helpers.logger.info("Data preperation complete.")

	def prepare_network(self):
		""" Builds the network.

		Replicates the networked outlined in the  Acute Leukemia Classification
		Using Convolution Neural Network In Clinical Decision Support System paper
		using Tensorflow 2.0.
		https://airccj.org/CSCP/vol7/csit77505.pdf
		"""

		self.val_steps = self.helpers.confs["train"]["val_steps"]
		self.batch_size = self.helpers.confs["train"]["batch"]
		self.epochs = self.helpers.confs["train"]["epochs"]

		self.tf_model = tf.keras.models.Sequential([
			tf.keras.layers.ZeroPadding2D(
				padding=(2, 2), input_shape=self.data.X_train.shape[1:]),
			tf.keras.layers.Conv2D(30, (5, 5), strides=1,
				padding="valid", activation='relu'),
			tf.keras.layers.ZeroPadding2D(padding=(2, 2)),
			tf.keras.layers.Conv2D(30, (5, 5), strides=1,
				padding="valid", activation='relu'),
			tf.keras.layers.MaxPooling2D(
				pool_size=(2, 2), strides=2, padding='valid'),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(2),
			tf.keras.layers.Activation('softmax')
		],
		"AllOneApiClassifier")
		self.tf_model.summary()

		self.helpers.logger.info("Network initialization complete.")

	def train(self):
		""" Trains the model

		Trains the neural network.
		"""

		self.helpers.logger.info("Using Adam Optimizer.")
		optimizer = tf.keras.optimizers.Adam(lr=self.helpers.confs["train"]["learning_rate_adam"],
										decay = self.helpers.confs["train"]["decay_adam"])

		self.tf_model.compile(optimizer=optimizer,
						loss='binary_crossentropy',
						metrics=[tf.keras.metrics.BinaryAccuracy(name='acc'),
								tf.keras.metrics.Precision(name='precision'),
								tf.keras.metrics.Recall(name='recall'),
								tf.keras.metrics.AUC(name='auc') ])

		self.history = self.tf_model.fit(self.data.X_train, self.data.y_train,
									validation_data=(self.data.X_test, self.data.y_test),
									validation_steps=self.val_steps,
									epochs=self.epochs)

		print(self.history)
		print("")

		self.freeze_model()
		self.save_model_as_json()
		self.save_weights()

	def freeze_model(self):
		""" Freezes the model """

		tf.saved_model.save(
			self.tf_model, self.helpers.confs["model"]["saved_model_dir"])

		fmodel = tf.function(lambda x: self.tf_model(x))
		fmodel = fmodel.get_concrete_function(
			x=tf.TensorSpec(self.tf_model.inputs[0].shape, self.tf_model.inputs[0].dtype))

		freeze = convert_variables_to_constants_v2(fmodel)
		freeze.graph.as_graph_def()

		layers = [op.name for op in freeze.graph.get_operations()]
		self.helpers.logger.info("Frozen model layers")
		for layer in layers:
			self.helpers.logger.info(layer)

		self.helpers.logger.info("Frozen model inputs")
		self.helpers.logger.info(freeze.inputs)
		self.helpers.logger.info("Frozen model outputs")
		self.helpers.logger.info(freeze.outputs)

		tf.io.write_graph(graph_or_graph_def=freeze.graph,
			logdir=self.helpers.confs["model"]["freezing_log_dir"],
			name=self.helpers.confs["model"]["frozen"],
			as_text=False)

	def save_model_as_json(self):
		""" Saves the model as JSON """

		with open(self.model_json, "w") as file:
			file.write(self.tf_model.to_json())

		self.helpers.logger.info("Model JSON saved " + self.model_json)

	def save_weights(self):
		""" Saves the model weights """

		self.tf_model.save_weights(self.weights_file)
		self.helpers.logger.info("Weights saved " + self.weights_file)

	def visualize_metrics(self):
		""" Visualize the metrics. """

		plt.plot(self.history.history['acc'])
		plt.plot(self.history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.ylim((0, 1))
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('model/plots/accuracy.png')
		plt.show()
		plt.clf()

		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('model/plots/loss.png')
		plt.show()
		plt.clf()

		plt.plot(self.history.history['auc'])
		plt.plot(self.history.history['val_auc'])
		plt.title('Model AUC')
		plt.ylabel('AUC')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('model/plots/auc.png')
		plt.show()
		plt.clf()

		plt.plot(self.history.history['precision'])
		plt.plot(self.history.history['val_precision'])
		plt.title('Model Precision')
		plt.ylabel('Precision')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('model/plots/precision.png')
		plt.show()
		plt.clf()

		plt.plot(self.history.history['recall'])
		plt.plot(self.history.history['val_recall'])
		plt.title('Model Recall')
		plt.ylabel('Recall')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('model/plots/recall.png')
		plt.show()
		plt.clf()

	def confusion_matrix(self):
		""" Prints/displays the confusion matrix. """

		self.matrix = confusion_matrix(self.data.y_test.argmax(axis=1),
								self.test_preds.argmax(axis=1))

		self.helpers.logger.info("Confusion Matrix: " + str(self.matrix))
		print("")

		plt.imshow(self.matrix, cmap=plt.cm.Blues)
		plt.xlabel("Predicted labels")
		plt.ylabel("True labels")
		plt.xticks([], [])
		plt.yticks([], [])
		plt.title('Confusion matrix ')
		plt.colorbar()
		plt.savefig('model/plots/confusion-matrix.png')
		plt.show()
		plt.clf()

	def figures_of_merit(self):
		""" Calculates/prints the figures of merit.

		https://homes.di.unimi.it/scotti/all/
		"""

		test_len = len(self.data.X_test)

		TP = self.matrix[1][1]
		TN = self.matrix[0][0]
		FP = self.matrix[0][1]
		FN = self.matrix[1][0]

		TPP = (TP * 100)/test_len
		FPP = (FP * 100)/test_len
		FNP = (FN * 100)/test_len
		TNP = (TN * 100)/test_len

		specificity = TN/(TN+FP)

		misc = FP + FN
		miscp = (misc * 100)/test_len

		self.helpers.logger.info(
			"True Positives: " + str(TP) + "(" + str(TPP) + "%)")
		self.helpers.logger.info(
			"False Positives: " + str(FP) + "(" + str(FPP) + "%)")
		self.helpers.logger.info(
			"True Negatives: " + str(TN) + "(" + str(TNP) + "%)")
		self.helpers.logger.info(
			"False Negatives: " + str(FN) + "(" + str(FNP) + "%)")

		self.helpers.logger.info("Specificity: " + str(specificity))
		self.helpers.logger.info("Misclassification: " +
						str(misc) + "(" + str(miscp) + "%)")

	def load(self):
		""" Loads the model """

		with open(self.model_json) as file:
			m_json = file.read()

		self.tf_model = tf.keras.models.model_from_json(m_json)
		self.tf_model.load_weights(self.weights_file)

		self.helpers.logger.info("Model loaded ")

		self.tf_model.summary()

	def evaluate(self):
		""" Evaluates the model """

		self.predictions()

		metrics = self.tf_model.evaluate(
			self.data.X_test, self.data.y_test, verbose=0)
		for name, value in zip(self.tf_model.metrics_names, metrics):
			self.helpers.logger.info("Metrics: " + name + " " + str(value))
		print()

		self.visualize_metrics()
		self.confusion_matrix()
		self.figures_of_merit()
		exit()

	def predictions(self):
		""" Gets a prediction for an image. """

		self.train_preds = self.tf_model.predict(self.data.X_train)
		self.test_preds = self.tf_model.predict(self.data.X_test)

		self.helpers.logger.info("Training predictions: " + str(self.train_preds))
		self.helpers.logger.info("Testing predictions: " + str(self.test_preds))

	def predict(self, img):
		""" Gets a prediction for an image. """

		predictions = self.tf_model.predict_proba(img)
		prediction = np.argmax(predictions, axis=-1)

		return prediction

	def reshape(self, img):
		""" Reshapes an image. """

		dx, dy, dz = img.shape
		input_data = img.reshape((-1, dx, dy, dz))

		return input_data

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

				start = time.time()
				img = cv2.imread(fileName).astype(np.float32)
				self.helpers.logger.info("Loaded test image " + fileName)

				img = cv2.resize(img, (self.helpers.confs["data"]["dim"],
									   self.helpers.confs["data"]["dim"]))
				img = self.reshape(img)

				prediction = self.predict(img)
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
