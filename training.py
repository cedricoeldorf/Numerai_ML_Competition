# -*- coding: utf-8 -*-
"""
Created on Sun Aug 07 18:20:33 2016

@author: Cedric Oeldorf
"""
path = "C:/Users/Cedric Oeldorf/Desktop/Projects/Numerai/numerai_training_data.csv"
path_t = "C:/Users/Cedric Oeldorf/Desktop/Projects/Numerai/numerai_tournament_data.csv"


import numpy
seed = 7
numpy.random.seed(seed)

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data(path):
    global X_train
    global y_train
    train = pd.read_csv(path)
    k = len(train.columns)    
    train = train.values    
    X_train = train[:,0:k-2]
    y_train = train[:,k-1]
    del train
    return X_train
    return y_train


    
load_data(path)


"""
Feature engineering 
"""



"""
Model
"""

kfold = StratifiedKFold(y=y_train, n_folds=10, shuffle=True, random_state=seed)
cvscores = []
for i, (train, test) in enumerate(kfold):
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=20, init='uniform', activation='relu'))
	model.add(Dense(10, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X_train, y_train, nb_epoch=150, batch_size=10, verbose=2)
	# evaluate the model
	scores = model.evaluate(X_train, y_train, verbose=2)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
 
print "%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores))