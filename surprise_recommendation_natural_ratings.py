# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 22:43:21 2018

@author: andrea
"""

import Dataset as ds
import pandas as pd
import numpy as np
import os.path
import sklearn as skl
from surprise import Dataset, Reader, SVD, BaselineOnly, accuracy
from surprise.dataset import DatasetUserFolds
from surprise.model_selection import GridSearchCV, cross_validate
from surprise.model_selection import split


dd = ds.Dataset()
dd.prune_ratings()
dd.prune_friends()
dd.split(test_ratio=0.17, seed=1)
dd.normalize_weights()

data = Dataset(Reader())
data.raw_ratings = [(uid, iid, r, None)
                                for (uid, iid, r) in
                                dd.ratings.itertuples(index=False)]

"""
# Global Mean prediction (2.71)
trainset, testset = split.train_test_split(data, test_size=.17, random_state=1)
rmse = np.sqrt(np.sum(list(zip(*testset))[2] - trainset.global_mean)**2 / len(test))
print('RMSE with global mean: ',rmse)
"""
#%% SCORE ON PREDEFINED DATA SPLIT (1.1269)
"""
# Import predefined trainset and testset
try:
    data = DatasetUserFolds(reader=Reader())
except:
    # We are forcing this class to build without some necessary parameter
    # so we need to skip the errors raised
    pass

raw_trainset = [(uid, iid, r, None)
                                for (uid, iid, r) in
                                dd.train.itertuples(index=False)]
raw_testset = [(uid, iid, r, None)
                                for (uid, iid, r) in
                                dd.test.itertuples(index=False)]
trainset = data.construct_trainset(raw_trainset)
testset = data.construct_testset(raw_testset)

algo = SVD(n_factors=5, n_epochs=30, lr_all=1.e-3, reg_all=1.e-4)
algo.fit(trainset)
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)

"""
# Set Grid Parameters (1.134)
param_grid = {
        'n_factors' : [5,10],
        'n_epochs' : [30,40],
        'lr_all' : np.logspace(-3.5,-1, 5),
        'reg_all' : np.logspace(-5,-3, 5),
#        'bsl_options' : {'method': ['als'],
#                         'reg_i': np.logspace(-5,0,10),
#                         'reg_u': np.logspace(-5,0,10),
#                         'n_epochs': [10]
#                        }
}

# Init grid_search
grid = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=6, n_jobs=1, joblib_verbose=10000)
grid.fit(data)

# Print best score and best parameters
print('Best Score: ', grid.best_score['rmse'])
print('Best parameters: ', grid.best_params['rmse'])