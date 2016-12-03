# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:12:17 2016

@author: Cedric Oeldorf
"""

est =100
lr = 0.05
rs =1
def random_forest(X,training_target,est,Y):
    from sklearn.ensemble import RandomForestClassifier
    global proba 
    global fi
    clf = RandomForestClassifier(n_estimators=est)
    clf = clf.fit(X,training_target)
    proba = clf.predict_proba(Y)
    fi = clf.feature_importances_
def adaboost(X,training_target,Y,est):

    from sklearn.ensemble import AdaBoostClassifier


    clf = AdaBoostClassifier(n_estimators=est)
    clf.fit(X,training_target)
    proba = clf.predict_proba(Y)
    
def gradient_boost(X,training_target,Y,est,lr,depth,rs):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=est, learning_rate=lr,
    random_state=rs).fit(X[:,[211,17,82,214,74,128,182,79,121,81,0,124,76,217,218,192,232,195,38]],training_target)
x2 = X[:,[211,17,82,214,74,128,182,79,121,81,0,124,76,217,218,192,232,195,38]]


    proba = clf.predict_proba(Y[:,[211,17,82,214,74,128,182,79,121,81,0,124,76,217,218,192,232,195,38]])

def voting_class(X,training_target,Y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    eclf.fit(X[:,0:6],training_target)
    proba = eclf.predict_proba(Y[:,0:6])
    
    eclf.predict()
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
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

X[:,[211,17,82,214,74,128,182,79,121,81,0,124,76,217,218,192,232,195,38]]
    
    
    
    
random_forest(X,training_target,100,Y)


adaboost(X,training_target,Y,est)
gradient_boost(X,training_target,Y,est,lr,depth,rs)
voting_class(X,training_target,Y)


stack = pd.DataFrame()
stack["grad"] = proba[:,1]
