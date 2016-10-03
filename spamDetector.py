"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import sys
import numpy as np
import download as dw
import dataExtraction as de
import naiveBayesClassifier as nbc
from sklearn.linear_model import LogisticRegression

def dataTrainTest(destination, dataSetName, saveName, testNumber):
	print("Getting data")
	dw.downloadAndExtractFile(destination)
	features, classes = de.openDataSet(dataSetName)

	print("Creates test and train sets")
	trainFeatures, trainClasses, testFeatures, testClasses = de.dataSeparation(
		features, classes, testNumber)

	print("Training")
	model, result = nbc.train(trainFeatures, trainClasses, saveName)
	print(str(result) +' / '+ str(trainFeatures.shape[0]))

	print("Testing")
	result = nbc.test(testFeatures, testClasses, model)
	print(str(result) +' / '+ str(testFeatures.shape[0]))

def help():
	print("spamDetector (-train [-t NumberOfTest] [-m modelSaveName]|-test [-m ModelFileName] FileName)")
	quit()

def main():
	arg = sys.argv
	saveName = "emails.model"
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
			else:
				help()
			print("NotYetImplemented")
	else:
		help()

if __name__ == '__main__':
	main()
