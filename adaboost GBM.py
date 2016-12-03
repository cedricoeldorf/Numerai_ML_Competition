# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:08:51 2016

@author: Cedric Oeldorf
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


"""
Predictions
"""
from keras.callbacks import History
hist = History()

model = Sequential()
model.add(Dense(21, input_dim=16, init='uniform', activation='relu'))
model.add(Dense(80, init='uniform', activation='relu'))
model.add(Dense(80, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
	# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


clf = GradientBoostingClassifier(n_estimators=100, verbose=2, learning_rate=0.05, max_depth=3, min_samples_leaf=1, random_state=1)
clf = RandomForestClassifier(n_estimators=100, verbose=2)
bdt = AdaBoostClassifier(base_estimator=clf, n_estimators=100)

bdt.fit(x2, training_target)


proba = bdt.predict_proba(y2)

ir = IsotonicRegression()

proba1 = ir.fit()
