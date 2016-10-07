"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import os
import sys
import numpy as np
import utils.download as dw
import utils.dataExtraction as de
import models.linearRegression as mod

def dataTrainTest(destination, dataSetName, saveName, testNumber):
	"""
	Downloads, creates a model and test it.
	"""
	print("Getting data")
	dw.downloadAndExtractFile(destination)
	features, classes = de.openDataSet(dataSetName)
	print('\t-> ' + str(classes.shape[0]) + ' documents in the dataset')

	print("Creates test and train sets")
	trainFeatures, trainClasses, testFeatures, testClasses = de.dataSeparation(
		features, classes, testNumber)

	print("Training")
	model, result = mod.train(trainFeatures, trainClasses, saveName)
	print('\t-> ' + str(round(100*result/trainFeatures.shape[0],2)) + ' / 100')

	print("Testing")
	result = mod.test(testFeatures, testClasses, model)
	print('\t-> ' + str(round(100*result/testFeatures.shape[0],2)) + ' / 100')

def addToDataSet(features, res, dataSet):
	answer = True
	result = 0
	while answer == True:
		print("Is this result correct ? (y/n)")
		answerText = input()
		if ("y" in answerText):
			answer = False
			result = int(res)
		elif ("n" in answerText):
			answer = False
			if not(res):
				result = 1
			else :
				result = 0
	while answer == False:
		print("Do you want to add this data in the database ? (y/n)")
		answerText = input()
		if ("y" in answerText):
			with open(dataSet, 'a') as dataBase:
				data = features.tolist()+[result]
				dataBase.write(','.join(map(str, data))+'\n')
			answer = True
		elif ("n" in answerText):
			answer = True

def testEmail(modelFileName, fileNames, dataSet):
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
			addToDataSet(features, res, dataSet)

def help():
	print("spamDetector (-train [-t NumberOfTest] [-m modelSaveName]|-test -m ModelFileName (-f FileName)*)")
	quit()

def main():
	arg = sys.argv
	saveName = mod.saveName
	destination = "DataSet"
	dataSet = destination + "/spambase.data"
	fileNames = []
	if len(arg) < 2:
		help()
	elif "-train" in arg[1]:
		if len(arg) % 2 == 0:
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
			testEmail(saveName, fileNames, dataSet)
		else:
			help()
	else:
		help()

if __name__ == '__main__':
	main()
