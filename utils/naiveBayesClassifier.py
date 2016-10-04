"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from math import exp, sqrt, log, pi

def train(trainingFeatures, trainingClasses, saveName):
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

	np.save(saveName, model)
	return model, test(trainingFeatures, trainingClasses, model)

def open(modelFileName):
	"""
	Opens the model in the given fileName
	"""
	try:
		return np.load(modelFileName)
	except Exception as e:
		quit()

def compute(features, model) :
	"""
	Computes the log(p(Spam|Email) / p(NonSpam|Email))
	Modelizes the p as a gaussian density
	"""
	res = 0
	for t in range(len(features)):
		feature = features[t]
		# Spam mean and var
		smean = model[0,t]
		svar = model[1,t]

		# Non Spam mean and var
		nmean = model[2,t]
		nvar = model[3,t]

		res += log(sqrt(nvar/svar)) + (-(feature-smean)**2/(2*svar)
			+ (feature-nmean)**2/(2*nvar))
	return res > 0

def test(testingFeatures, testingClasses, model):
	"""
	Tests the different given emails
	"""
	wellRecognized = 0
	for c in range(len(testingClasses)):
		res = compute(testingFeatures[c], model)
		if res and testingClasses[c] == 1:
			wellRecognized += 1
		elif not(res) and testingClasses[c] == 0:
			wellRecognized += 1
	return wellRecognized
