"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from math import exp, sqrt, log, pi
from .model import model

class linearRegression(model) :
	saveName = "emails.lr.npy"

	def train(self, trainingFeatures, trainingClasses):
		"""
		Computes a linear model Y = XB
		B = (Xt*X)^(-1)*Xt*Y
		B = A     ^(-1)*Xt*Y
		"""
		A = np.matmul(np.transpose(trainingFeatures),trainingFeatures)
		inter = np.matmul(np.linalg.inv(A),np.transpose(trainingFeatures))
		self.model = np.matmul(inter, trainingClasses)

		np.save(self.saveName, self.model)
		return self.test(trainingFeatures, trainingClasses)

	def compute(self, features) :
		"""
		Computes the log(p(Spam|Email) / p(NonSpam|Email))
		Modelizes the p as a gaussian density
		"""
		res = np.matmul(features, self.model)
		return res > 0.5
