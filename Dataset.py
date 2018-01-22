# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 00:04:34 2018

@author: andrea
"""
import os
import pandas as pd
import numpy as np
import networkx as nx
    
class Dataset():
    """ This class contains the dataset and functions to elaborate it """
    
    def __init__(self, folder_path=os.path.join('.','data')):
        # Define paths of the files
        artists_path = os.path.join(folder_path,'artists.dat')
        friends_path = os.path.join(folder_path, 'user_friends.dat')
        ratings_path = os.path.join(folder_path,'user_artists.dat')
        tags_path = os.path.join(folder_path,'tags.dat')
        tags_assign_path = os.path.join(folder_path,'user_taggedartists-timestamps.dat')
        
        # Import data
        self.artists = pd.read_csv(artists_path, sep='\t', header=0, index_col=0, skipinitialspace=True)
        self.friends = pd.read_csv(friends_path, sep='\t', header=0, index_col=0, skipinitialspace=True)
        self.ratings = pd.read_csv(ratings_path, sep='\t', header=0, skipinitialspace=True)
        self.tags = pd.read_csv(tags_path, sep='\t', header=0, index_col=0, skipinitialspace=True, encoding='latin1')
        self.tags_assign = pd.read_csv(tags_assign_path, sep='\t', header=0, skipinitialspace=True)
        # Define variables that contains list of users ID
        self.users = sorted(list(set(self.ratings.userID)))
        
        # Inizialize train and test
        self.train = self.ratings
        self.test = pd.DataFrame()
        
    @property
    def nuser(self):
        return len(self._userID2POS)
    @property
    def nart(self):
        return len(self.artists)
    @property
    def ntag(self):
        return len(self.tags)
        
    def prune_ratings(self, max_weight=50000, min_nart=49):
        """ Drop users considering the weights """
        
        users_to_drop = set()

        # Group ratings based on users
        group = self.ratings.groupby('userID')
        
        # Drop users with too high max weight (looking at the distribution
        # they seems to be outliers)
        d = group.max()
        users_to_drop.update(d[d.weight > max_weight].index)
        
        # Drop users with few artists
        d = group.nunique().artistID
        users_to_drop.update(d[d < min_nart].index)
        
        # Drop users from all the data
        self.drop_users(users_to_drop)
        print(len(users_to_drop), ' users dropped in weights pruning')
        
    def prune_friends(self, min_conn=0):
        """ Drop users considering the social network """
        # Build social network graph
        G = nx.Graph(self.build_friend_friend())
        # Extract biggest connected components
        G = next(nx.connected_component_subgraphs(G))
        users_to_drop = {u for u in self.users 
                         if self.get_userPOS(u) not in list(G.node)}
        # delete all nodes (users) with less than min_conn connections
        degree = dict(G.degree())
        remove = [self.get_userID(user) for user,degree in degree.items() if degree < min_conn]
        users_to_drop.update(remove)
        self.drop_users(users_to_drop)
        
        print(len(users_to_drop), ' users dropped in friendship pruning')
        
    def drop_users(self, users_to_drop):
        """ Drop given users from all the dataset """
        
        self.ratings = self.ratings[~ self.ratings.userID.isin(users_to_drop)]
        self.friends.drop(users_to_drop, inplace=True)
        self.friends = self.friends[~ self.friends.friendID.isin(users_to_drop)]
        self.tags_assign = self.tags_assign[~ self.tags_assign.userID.isin(users_to_drop)]
        self.users = [i for i in self.users if i not in users_to_drop]
        
    def normalize_weights(self):
        # Normalize weights for each user
        #group = self.ratings[['userID', 'weight']].groupby('userID')
        #tots = group.sum().weight.to_dict()
        #self.ratings.weight = self.ratings.weight / [tots[u] for u in self.ratings.userID]
        
        # Extract ratings based on quartiles for all ratings
        group = self.ratings.groupby('userID').weight
        # Sort ratings from 1 to 5
        self.ratings.weight = group.rank() / [l for n in group.size() for l in [n]*n] * 4 + 1
        # Normalize test ratings using the updated weights
        self.test = self.ratings.iloc[self.ratings.index.isin(self.test.index)]
        
        # Extract ratings based on quartiles for the train ratings
        group = self.train.groupby('userID').weight
        # Sort ratings from 1 to 5
        self.train.weight = group.rank() / [l for n in group.size() for l in [n]*n] * 4 + 1
        
    def init_POS_ID_translator(self):
        """ Initialize ID translator dictionaries:
            # ID: user and artists ID as they compare on the dataset
            # POS: unique sorted user/artist identifier (no holes),
                    to use for example in matrices 
        """
        self._artistID2POS = {i:p for p,i in enumerate(self.artists.index)}
        self._artistPOS2ID = {p:i for p,i in enumerate(self.artists.index)}
        self._userID2POS = {i:p for p,i in enumerate(self.users)}
        self._userPOS2ID = {p:i for p,i in enumerate(self.users)}
        
        
    def build_art_user(self, train_only=True):
        """ Build artist_user adjacency matrix using weights """
        # If ID translator was not inizialized, ask for it
        if not self._artistID2POS:
            raise ValueError('Inizialize POS_ID translator before')
            
        art_user = np.zeros((self.nart, self.nuser))
        # Choose as iterator all the ratings or only the ratings in the train
        if train_only:
            iterator = self.train
        else:
            iterator = self.ratings
        
        # Build matrix
        for index, row in iterator.iterrows():
            apos = self.get_artistPOS(row.artistID)
            upos = self.get_userPOS(row.userID)
            art_user[apos,upos] = row.weight
            
        return art_user
    
    def split(self, test_ratio=0.2, seed=None):
        """ Split data in trainset and testset """
        N = len(self.ratings)
        shuffled = self.ratings.sample(frac=1, random_state=seed)
        self.train = shuffled.iloc[: round(N*(1-test_ratio))]
        self.test = shuffled.iloc[round(N*(1-test_ratio)) :]
        
    
    def build_friend_friend(self):
        """ Build friend_friend matrix using social network connections """
        
        # If ID translator was not inizialized, ask for it
        if not self._artistID2POS:
            raise ValueError('Inizialize POS_ID translator before')
        
        friend_friend = np.zeros((self.nuser, self.nuser))
        for index, row in self.friends.iterrows():
            upos1 = self.get_userPOS(index)
            upos2 = self.get_userPOS(row.friendID)
            friend_friend[upos1,upos2] = 1
            
        # Symmetrize matrix if it is not
        friend_friend = np.where(friend_friend, friend_friend, friend_friend.T)
            
        return friend_friend
        
    def get_artistPOS(self, ID):
        # If ID translator was not inizialized, ask for it
        if not self._artistID2POS:
            raise ValueError('Inizialize POS_ID translator before')
        return self._artistID2POS[ID]

    def get_artistID(self, POS):
        # If ID translator was not inizialized, ask for it
        if not self._artistID2POS:
            raise ValueError('Inizialize POS_ID translator before')
        return self._artistPOS2ID[POS]
    
    def get_userPOS(self, ID):
        # If ID translator was not inizialized, ask for it
        if not self._artistID2POS:
            raise ValueError('Inizialize POS_ID translator before')
        return self._userID2POS[ID]

    def get_userID(self, POS):
        # If ID translator was not inizialized, ask for it
        if not self._artistID2POS:
            raise ValueError('Inizialize POS_ID translator before')
        return self._userPOS2ID[POS]