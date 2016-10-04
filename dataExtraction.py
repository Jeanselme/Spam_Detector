"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import pandas as pd
import numpy as np

FEATURES_NUMBER = 57

def openDataSet(dataSetName):
	"""
	Opens the dataset and extracts classes and emails features
	"""
	emails = pd.read_table(dataSetName, header=None, sep=',')
	emails_classes = emails[FEATURES_NUMBER].as_matrix()
	emails_features = emails.drop([FEATURES_NUMBER], axis=1).as_matrix()

	return emails_features, emails_classes

def randomizeData(features, classes):
	"""
	Creates a randomized permutation of the dataSet
	"""
	permutation = np.random.permutation(classes.shape[0])
	shuffled_classes = classes[permutation]
	shuffled_features = features[permutation,:]

	return shuffled_features, shuffled_classes

def dataSeparation(features, classes, testNumber):
	"""
	Creates the two datasets for tests and training
	"""
	features, classes = randomizeData(features, classes)

	testFeatures = features[:testNumber]
	testClasses = classes[:testNumber]
	trainingFeatures = features[testNumber:]
	trainingClasses = classes[testNumber:]

	return trainingFeatures, trainingClasses, testFeatures, testClasses

def emailToVector(fileName):
	"""
	Computes the same feature than in the dataset
	"""
	print("Email features extraction not implemented")
	res = np.zeros(FEATURES_NUMBER)
	with open(fileName, 'r') as email:
		pass
	return res
