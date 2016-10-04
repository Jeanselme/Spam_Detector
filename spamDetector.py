"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import os
import sys
import numpy as np
import download as dw
import dataExtraction as de
import naiveBayesClassifier as mod

def dataTrainTest(destination, dataSetName, saveName, testNumber):
	"""
	Downloads, creates a model and test it.
	"""
	print("Getting data")
	dw.downloadAndExtractFile(destination)
	features, classes = de.openDataSet(dataSetName)

	print("Creates test and train sets")
	trainFeatures, trainClasses, testFeatures, testClasses = de.dataSeparation(
		features, classes, testNumber)

	print("Training")
	model, result = mod.train(trainFeatures, trainClasses, saveName)
	print('\t-> ' + str(round(100*result/trainFeatures.shape[0],2)) + ' / 100')

	print("Testing")
	result = mod.test(testFeatures, testClasses, model)
	print('\t-> ' + str(round(100*result/testFeatures.shape[0],2)) + ' / 100')

def testEmail(modelFileName, fileNames):
	"""
	Tests text emails
	"""
	if not(os.path.exists(modelFileName)):
		print("Model does not exist")
	model = mod.open(modelFileName)
	for fileName in fileNames:
		if not(os.path.exists(fileName)):
			print("Email does not exist")
		else :
			features = de.emailToVector(fileName)
			res = mod.compute(features, model)

			if res :
				print("This mail is categorized as a spam")
			else :
				print("This mail is categorized as a safe email")

def help():
	print("spamDetector (-train [-t NumberOfTest] [-m modelSaveName]|-test -m ModelFileName (-f FileName)*)")
	quit()

def main():
	arg = sys.argv
	saveName = "emails.model.npy"
	fileNames = []
	if len(arg) < 2:
		help()
	elif "-train" in arg[1]:
		if len(arg) % 2 == 0:
			destination = "DataSet"
			dataSet = destination + "/spambase.data"
			testNumber = 100
			i = 2
			while i+1 < len(arg):
				if arg[i] == "-t":
					testNumber = int(arg[i+1])
					i+=2
				elif arg[i] == "-m":
					saveName = arg[i+1]
					i+=2
				else:
					help()
			dataTrainTest(destination, dataSet, saveName, testNumber)
		else :
			help()
	elif "-test" in arg[1] :
		i = 2
		while i+1 < len(arg):
			if arg[i] == "-m":
				saveName = arg[i+1]
				i+=2
			elif arg[i] == "-f":
				fileNames.append(arg[i+1])
				i+=2
			else:
				help()
		if fileNames != []:
			testEmail(saveName, fileNames)
		else:
			help()
	else:
		help()

if __name__ == '__main__':
	main()
