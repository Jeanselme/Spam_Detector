"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

class model(object):
	"""
	Mother class for all model
	"""

	def train(self, trainingFeatures, trainingClasses):
		print("Should be implemented in child classes")

	def compute(self, features):
		print("Should be implemented in child classes")

	def open(self):
		"""
		Opens the model in the given fileName
		"""
		try:
			self.model = np.load(saveName)
		except Exception as e:
			quit()

	def test(self, testingFeatures, testingClasses):
		"""
		Tests the different given emails
		"""
		wellRecognized = 0
		for c in range(len(testingClasses)):
			res = self.compute(testingFeatures[c])
			if res and testingClasses[c] == 1:
				wellRecognized += 1
			elif not(res) and testingClasses[c] == 0:
				wellRecognized += 1
		return wellRecognized
