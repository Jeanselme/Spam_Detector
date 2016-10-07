# Spam_Detector
A simple example to apply machine learning algorithm to an everyday used algorithm

## DataSet
This algorithm uses the dataset extracted from the website of Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

## Naive Bayes Classifier
The Naive Bayes Classifier is a probabilistic classifier which assumes that all features are independent.  
This algorithm is based on :

### Bayes rule
```
P(Spam|Email) = P(Spam) * P(Email | Spam) / P(Email)
```

### Modeling
To model the P(Email = (features(i)), Spam) = Sum(P(features(i)|Spam)) * P(Spam),  
the current version uses a gaussian distibution for each feature with a mean and a variance given by the training set.

### Decision
In order to decide if an email is a spam or not, the program computes ln(P(Spam|Email)/P(NonSpam|Email)). If the result is positive, the email is classified as a spam, otherwise it is not a spam.

### Result

## Linear Regression
Computes a simple linear regression on the model

### Result

## Execution
In order to change the model, go in spamDetector and change the library called.
For the learning phasis :  
```
python3.5 spamDetector.py -train
```

For the training phasis :  
```
python3.5 spamDetector.py -test -f emailName
```

## Libraries
Needs numpy, os, pandas, sys, urllib.request and zipfile. Executed with python3.5
