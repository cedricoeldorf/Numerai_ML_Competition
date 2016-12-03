# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 22:41:25 2016

@author: Cedric Oeldorf
"""

class feature_engineering:
    def read_data(path):
        import pandas as pd
        import numpy as np
        global all
        orig = pd.read_csv(path)
        all_dat = orig.drop(["validate", "target"], axis=1)
        all_dat = np.array(all_dat).astype(np.float32)
        
    def pca(data,components):
        
        from sklearn.decomposition import PCA
        global pca_transf
        pca = PCA(n_components=components)
        pca.fit(data)
        pca_transf = pca.fit_transform(data)
        print(pca.explained_variance_ratio_) 
        pca.score(data)
        
    def kmeans(data, clusters):
        from sklearn import cluster, datasets
        from pandas import pd        
        global k_means_transf
        k_means = cluster.KMeans(n_clusters=clusters)
        k_means_transf = k_means.fit(data) 
        k_means_transf = pd.DataFrame(data=k_means_transf)

    def multiply(data):
        global multi
        import pandas as pd
        multi = pd.Dataframe()
        features = list(data.columns)
        for f in features:
            for g in features:
                if f != g:
                    if not (str(g) + "_" + str(f)) in data.columns:
                        multi[str(f) + "_" + str(g)] = data[f] * data[g]
                        
    def polynomials(data):
        from sklearn import preprocessing
        global poly
        poly = PolynomialFeatures(interaction_only=True)
        poly = poly.fit_transform(data)
    
    def agglo(data):
        global agglo
        import pandas as pd
        from sklearn import cluster
        agglo = cluster.FeatureAgglomeration(n_clusters=32)

        agglo.fit(data)
        agglo = agglo.transform(data)
        agglo = pd.DataFrame(data=agglo)
    
    def combine_eng(pca,kmeans,multi,poly,agglo,orig):
        global X_train
        global tournament
        global y_train
        import pandas as pd
        import numpy as np
        l_X = pd.concat([pca, kmeans, multi,poly,agglo,orig], axis=1, join='inner')
        training = l_X[l_X["validate"] == 0]
        validation = l_X[l_X["validate"] == 1]
        training = training.drop("validate", axis=1)
        validation = validation.drop("validate", axis=1)
        y_train = training["target"].values.T.astype(np.int32)
        training = training.drop("target", axis=1)
        validation = validation.drop("target", axis=1)
        X_train = np.array(training).astype(np.float32)
        tournament = np.array(validation).astype(np.float32)
