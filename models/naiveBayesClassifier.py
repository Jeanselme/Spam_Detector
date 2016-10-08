"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from math import exp, sqrt, log, pi
from .model import model

class naiveBayesClassifier(model):
	saveName = "emails.nbc.npy"

	def train(self, trainingFeatures, trainingClasses):
		"""
		Creates a matrix of mean and variance given a class
		"""
		spam = np.where(trainingClasses == 1)
		nonspam = np.where(trainingClasses == 0)

		model = np.zeros((4, trainingFeatures.shape[1]))
		# Computes the mean, variance, min and max of the spam on each feature
		model[0] = np.mean(trainingFeatures[spam,:], axis=1)
		model[1] = np.var(trainingFeatures[spam,:], axis=1)

		# Computes the mean and variance of the nonspam on each feature
		model[2] = np.mean(trainingFeatures[nonspam,:], axis=1)
		model[3] = np.var(trainingFeatures[nonspam,:], axis=1)

		self.model = model
		np.save(self.saveName, model)
		return self.test(trainingFeatures, trainingClasses)

	def compute(self, features) :
		"""
		Computes the log(p(Spam|Email) / p(NonSpam|Email))
		Modelizes the p as a gaussian density
		"""
		res = 0
		for t in range(len(features)):
			feature = features[t]
			# Spam mean and var
			smean = self.model[0,t]
			svar = self.model[1,t]

			# Non Spam mean and var
			nmean = self.model[2,t]
			nvar = self.model[3,t]

			res += log(sqrt(nvar/svar)) + (-(feature-smean)**2/(2*svar)
				+ (feature-nmean)**2/(2*nvar))
		return res > 0
