"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import os
import zipfile
from urllib.request import urlretrieve

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/"
file = "spambase.zip"

def extract(fileName, destination) :
	"""
	Extract the content of the document
	"""
	if not(os.path.exists(destination)) :
		with zipfile.ZipFile(fileName, "r") as zipFile:
			print("Extraction")
			zipFile.extractall(destination)
			print("Extraction Complete")
	else :
		print("Destination directory already exists")

def downloadAndExtractFile(destination, fileName = file, adress = url, force = False) :
	"""
	Downloads the file at the given adress and saves it under fileName
	"""
	if not(os.path.exists(fileName)) or force :
		print("Download")
		urlretrieve (adress + fileName, fileName)
		print("Download Complete")
		extract(fileName, destination)
	else :
		print("File already present on your system")
