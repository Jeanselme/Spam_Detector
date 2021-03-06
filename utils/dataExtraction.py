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

def countWords(text, words):
	"""
	Counts the occurence of the list of words and returns a list of number
	"""
	res = np.zeros(len(words))
	textList = text.split()
	for word in range(len(words)):
		for presentWord in textList:
			if words[word] in presentWord :
				res[word] += 100/len(textList)
	return res

def countChar(text, chars):
	"""
	Counts the occurence of the list of characters and returns a list of number
	"""
	res = np.zeros(len(chars))
	textList = text.split()
	charText = []
	for word in textList:
		charText += list(word)
	for char in range(len(chars)):
		res[char] += 100 * textList.count(chars[char])/len(chars)
	return res

def capitalLetter(text):
	"""
	Extracts the features of capital letter
	"""
	res = [0,0,0]
	stop = True
	cap = []
	textList = text.split()
	charText = []
	for word in textList:
		charText += list(word)
	for char in charText:
		if (char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
			res[2] += 1
			if stop == True:
				cap.append(char)
			else:
				cap[len(cap)-1] += char
			stop = False
		else:
			stop = True
	if cap != []:
		res[1] = max([len(cap[i]) for i in range(len(cap))])
		res[0] = res[2] / len(cap)
	return res

def emailToVector(fileName):
	"""
	Computes the same feature than in the dataset
	"""
	res = np.zeros(FEATURES_NUMBER)
	with open(fileName, 'r') as email:
		text = email.read()
		res[0:48] = countWords(text,["make","address","all","3d","our","over",
			"remove","internet","order","mail","receive","will","people","report",
			"addresses","free","business","email","you","credit","your","font",
			"000","money","hp","hpl","george","650","lab","labs","telnet","857",
			"data","415","85","technology","1999","parts","pm","direct","cs",
			"meeting","original","project","re","edu","table","conference"])
		res[48:54] = countChar(text,[';','(','[','!','$','#'])
		res[54:] = capitalLetter(text)
	return res
