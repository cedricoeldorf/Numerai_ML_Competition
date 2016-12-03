# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 23:47:32 2016

@author: Cedric Oeldorf

GBM
"""

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, min_samples_leaf=1, random_state=1)
clf.fit(x2,training_target)
x2 = X[:,[259,214,17,76,82,99,211,74,121,1,256,81,79,128,193,206]]
y2 = Y[:,[259,214,17,76,82,99,211,74,121,1,256,81,79,128,193,206]]

proba = clf.predict_proba(y2)

importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]


    # Print the feature ranking
print("Feature ranking:")    
for f in range(x2.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))  
    # Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x2.shape[1]), importances[indices],
    color="r", align="center")
plt.xticks(range(x2.shape[1]), indices)
plt.xlim([-1, x2.shape[1]])
plt.show()  

feature_imp(X)
clf.get_params()

param_grid = [
  {'learning_rate': [0.05, 0.1, 0.2, 0.25], 'max_depth': [3,4,5,6], 'min_samples_leaf': [1,2], 'n_estimators': [100,200,300]},
 ]
 
svr = GradientBoostingClassifier() 
from sklearn import grid_search
clf = grid_search.GridSearchCV(svr, param_grid)
clf.fit(x2,training_target)


    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()