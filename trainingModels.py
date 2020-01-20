from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.linear_model import Perceptron as Perc
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import pickle
import random
import os
import warnings
import sys


def calculateScores(label, prediction):
	tp,tn,fp,fn = 0,0,0,0
	for i in range(len(prediction)):
		#print(label[i], prediction[i])
		if(label[i] == prediction[i]):
			if(label[i] == 0):
				tn += 1
			else:
				tp += 1
		else:
			if(label[i] ==0):
				fp += 1
			else:
				fn += 1
	if(tp+fp==0 or tp+fn==0):
		return [-1, -1, -1, -1]
	precision, recall = tp/(tp+fp), tp/(tp+fn)
	if(precision+recall==0):
		return [-1, -1, -1, -1]
	F1, accuracy = 2*precision*recall/(precision+recall), (tp+tn)/(tp+tn+fp+fn)
	#print("Confusion mertix: \n", confusion_matrix(label, prediction))
	return [precision, recall, F1, accuracy]

def calculateAverageScores(D):
	precision = {'NeuralNetwork': [], 'RF': [], 'SVM': [], 'KNN': [], 'NB': [],'GB': [], 'XGB': [], 'LR': []}
	recall = {'NeuralNetwork': [], 'RF': [], 'SVM': [], 'KNN': [], 'NB': [],'GB': [], 'XGB': [], 'LR': []}
	F1= {'NeuralNetwork': [], 'RF': [], 'SVM': [], 'KNN': [], 'NB': [],'GB': [], 'XGB': [], 'LR': []}
	accuracy = {'NeuralNetwork': [], 'RF': [], 'SVM': [], 'KNN': [], 'NB': [],'GB': [], 'XGB': [], 'LR': []}
	if type(D) is list:
		pass
	else:
		D = [D]
	for d in D:
		#print(d)
		for key,v in d.items():
			precision[key].append(d[key][0])
			recall[key].append(d[key][1])
			F1[key].append(d[key][2])
			accuracy[key].append(d[key][3])
	for key in accuracy.keys():
		precision[key]  = sum(precision[key])/len(precision[key])
		recall[key]  = sum(recall[key])/len(recall[key])
		F1[key]  = sum(F1[key])/len(F1[key])
		accuracy[key]  = sum(accuracy[key])/len(accuracy[key])

	return [precision, recall, F1, accuracy]

def SVM(trainData, trainLable):
	clf = svm.SVC(kernel='poly', C=0.5)
	clf.fit(trainData, trainLable)
	return clf

def knn(trainData, trainLable, n_neighbors):
	clf = neighbors.KNeighborsClassifier(n_neighbors)
	clf.fit(trainData, trainLable)
	return clf

def GNB(trainData, trainLable):
	clf = GaussianNB()
	clf.fit(trainData, trainLable)
	return clf

def XGBoosting(trainData, trainLable):
	clf = XGBClassifier()
	clf.fit(trainData, trainLable)
	return clf

def gradientBoosting(trainData, trainLable):
	clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=2, max_depth=2, random_state=0)
	clf.fit(trainData, trainLable)
	return clf

def randomForest(trainData, trainLable):
	clf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
	clf.fit(trainData, trainLable)
	return clf

def logisticRegression(trainData, trainLable):
	clf = LogisticRegression(random_state=0, solver='lbfgs').fit(trainData, trainLable)
	return clf

def neuralNetwork(trainData, trainLable):
	model = Sequential()
	model.add(Dense(12, input_dim=len(trainData[0]), activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(np.array(trainData), np.array(trainLable), epochs=200, batch_size=10, verbose=0)
	return model

## trains 'NeuralNetwork','RF','SVM','KNN','NB','GB','XGB','LR' and returns a dictionary d['NN']=Score
def trainAllModels(trainData, trainLable, testData, testLable):
	d = {}

	#print("LR | ")
	lrModel = logisticRegression(trainData, trainLable)
	predict = lrModel.predict(testData)
	lrScore = calculateScores(testLable, predict)
	d['LR'] = lrScore
	
	#print("RF | ")
	rfModel = randomForest(trainData, trainLable)
	predict = rfModel.predict(testData)
	randomForestScore = calculateScores(testLable, predict)
	d['RF'] = randomForestScore

	#print("Neural Net | ")
	nnModel = neuralNetwork(trainData, trainLable)
	predict = nnModel.predict(testData)
	nnScore = calculateScores(testLable, predict)
	d['NeuralNetwork'] = nnScore

	#print("SVM | ")
	svmModel = SVM(trainData, trainLable)
	predict = svmModel.predict(testData)
	svmScore = calculateScores(testLable, predict)
	d['SVM'] = svmScore

	#print("KNN | ")
	knnModel = knn(trainData, trainLable,5)
	predict = knnModel.predict(testData)
	knnScore = calculateScores(testLable, predict)
	d['KNN'] = knnScore

	#print("GNB | ")
	gnbModel = GNB(trainData, trainLable)
	predict = gnbModel.predict(testData)
	nbScore = calculateScores(testLable, predict)
	d['NB'] = nbScore

	#print("GB | ")
	gbModel = gradientBoosting(trainData, trainLable)
	predict = gbModel.predict(testData)
	gbScore = calculateScores(testLable, predict)
	d['GB'] = gbScore

	#print("XGB | ")
	xgModel = XGBoosting(trainData, trainLable)
	predict = xgModel.predict(testData)
	predict = [round(value) for value in predict]
	xgScore = calculateScores(testLable, predict)
	d['XGB'] = xgScore

	return d