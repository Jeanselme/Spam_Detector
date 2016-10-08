"""
	Spam Detector
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

from .linearRegression import linearRegression
from .naiveBayesClassifier import naiveBayesClassifier

class classifier:
	"""
	Class which returns the demanded model
	It uses the factory model
	"""

	def get(self, name) :
		name = name.lower()
		if name == "lr" :
			return linearRegression()
		elif name == "nbc" :
			return naiveBayesClassifier()
		else :
			print("Asked model is not yet implemented")
