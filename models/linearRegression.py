"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from math import exp, sqrt, log, pi

saveName = "emails.model.skl"

def train(trainingFeatures, trainingClasses, saveName):
	"""
	Computes a linear model Y = XB
	B = (Xt*X)^(-1)*Xt*Y
	B = A     ^(-1)*Xt*Y
	"""
	A = np.matmul(np.transpose(trainingFeatures),trainingFeatures)
	model = np.matmul(np.linalg.inv(A),np.transpose(trainingFeatures))
	model = np.matmul(model, trainingClasses)

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
	res = np.matmul(features, model)
	return res > 0.5

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
