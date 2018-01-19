# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 00:04:34 2018

@author: andrea
"""
import os
import pandas as pd
import numpy as np
    
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
        
        # Build ID translator dictionaries
        # ID: user and artists ID as they compare on the dataset
        # POS: unique sorted user/artist identifier (no holes), to use for example in matrices
        self._artistID2POS = {i:p for p,i in enumerate(self.artists.index)}
        self._artistPOS2ID = {p:i for p,i in enumerate(self.artists.index)}
        self._userID2POS = {i:p for p,i in enumerate(set(self.ratings.userID))}
        self._userPOS2ID = {p:i for p,i in enumerate(set(self.ratings.userID))}
        
    @property
    def nuser(self):
        return len(self._userID2POS)
    @property
    def nart(self):
        return len(self.artists)
    @property
    def ntag(self):
        return len(self.tags)
        
    def prune_ratings(self, max_weight=200000, min_nart=10):
        """ Drop users considering the weights """
        
        users_to_drop = set()

        # Group ratings based on users
        group = self.ratings.groupby('userID')
        
        # Drop users with too high max weight (looking at the distribution
        # they seems to be outliers)
        data = group.max()
        users_to_drop.update(data[data.weight > 200000].index)
        
        # Drop users with few artists
        data = group.nunique().artistID
        users_to_drop.update(data[data < 10].index)
        
        # Drop users from all the data
        self.drop_users(users_to_drop)
        print(len(users_to_drop), ' users dropped')
        
    def drop_users(self, users_to_drop):
        """ Drop given users from all the dataset """
        
        self.ratings = self.ratings[~ self.ratings.userID.isin(users_to_drop)]
        self.friends.drop(self.friends.index[list(users_to_drop)])
        self.friends = self.friends[~ self.friends.isin(users_to_drop)]
        self.tags_assign = self.tags_assign[~ self.tags_assign.userID.isin(users_to_drop)]
        
        # Update ID translator dictionaries
        self._userID2POS = {i:p for p,i in enumerate(set(self.ratings.userID))}
        self._userPOS2ID = {p:i for p,i in enumerate(set(self.ratings.userID))}
        
    def normalize_weights(self):
        # Normalize weights for each user
        group = self.ratings[['userID', 'weight']].groupby('userID')
        tots = group.sum().weight.to_dict()
        self.ratings.weight = self.ratings.weight / [tots[u] for u in self.ratings.userID]
        
    def build_art_user(self):
        """ Build artist_user adjacency matrix using weights """
        art_user = np.zeros((self.nart, self.nuser))
        for index, row in self.ratings.iterrows():
            apos = self.get_artistPOS(row.artistID)
            upos = self.get_userPOS(row.userID)
            art_user[apos,upos] = row.weight
            
        return art_user
    
    def build_friend_friend(self):
        """ Build friend_friend matrix using social network connections """
        friend_friend = np.zeros((self.nuser, self.nuser))
        for index, row in self.friends.iterrows():
            upos1 = self.get_userPOS(index)
            upos2 = self.get_userPOS(row.friendID)
            friend_friend[upos1,upos2] = 1
            
        return friend_friend
        
    def get_artistPOS(self, ID):
        return self._artistID2POS[ID]

    def get_artistID(self, POS):
        return self._artistPOS2ID[POS]
    
    def get_userPOS(self, ID):
        return self._userID2POS[ID]

    def get_userID(self, POS):
        return self._userPOS2ID[POS]