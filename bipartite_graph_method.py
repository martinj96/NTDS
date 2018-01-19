# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:44:33 2018

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import pandas as pd
import os.path
import networkx as nx
import scipy.sparse as sp
import sklearn as skl
import pygsp as gsp
from Dataset import Dataset

plt.rcParams['figure.figsize'] = (10, 5)
gsp.plotting.BACKEND = 'matplotlib'

#%% IMPORT DATA
data = Dataset()
data.prune_ratings()
data.normalize_weights()

#%% CONSTANTS
TRESHOLD = 5    # Treshold of user_user connection strength to keep

#%% BUILD ARTIST GRAPH EXAMPLE
# Build example graph
G = gsp.graphs.Community(N=data.nart, Nc=20, seed=42)
art_tags = G.W.todense()

# Use the cosine distance
distances = sci.spatial.distance.pdist(art_tags, metric='cosine')
distances = sci.spatial.distance.squareform(distances)

# Add simplest kernel to go from distance to similarity measure
art_art = 1 - distances

# Keep diagonal to consider also self connections

#%% CREATE USER NETWORKS

# Build art_user matrix
print('Build art_user matrix')
art_user = data.build_art_user()
    
#%% RUN ALGO

# Initialize user_user network
user_user = np.zeros((data.nuser,data.nuser))

# Number of neighbors to take for each artist
k = 2

for u,col in enumerate(art_user.T):
    print('User #', u)
    for source_ind in np.nonzero(col)[0]:
        source_weight = col[source_ind]
        for neighbor_ind in art_art[source_ind].argsort().T[-k:]:
            for new_user_ind in np.nonzero(art_user[np.asscalar(neighbor_ind)])[0]:
                neigh_weight = art_user[neighbor_ind, new_user_ind]
                similarity = 1 - np.abs(neigh_weight - source_weight)
                user_user[u, new_user_ind] += similarity
                
# Clear connection of user with itself
np.fill_diagonal(user_user, 0)
# Prune user_user connections depending on the entries value
user_user[user_user < TRESHOLD] = 0
user_user[user_user >= TRESHOLD] = 1
# Make the matrix symmetric
user_user = np.where(user_user>0, user_user, user_user.T)

#%% NETWORKS COMPARISON

# Build friend_friend matrix
friend_friend = data.build_friend_friend()