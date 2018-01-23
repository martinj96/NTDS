# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt

def plot_weight_distribution(weights):
	ax = plt.figure().gca()
	weights.hist(ax=ax, bins=30, log=True)
	ax.set_title('Weigths Distribution')
	ax.set_xlabel('weigth')

def plot_statistics_on_user(users, drop=False):
	if drop:
		users_to_drop = set()

	_, [ax1,ax2,ax3] = plt.subplots(3,1, figsize=(20,10))

	# Plot max
	data = users.max()
	data.hist(ax=ax1, column='weight', bins=30, log=True)
	ax1.set_title('Max weigth distribution over users')
	ax1.set_xlabel('weigth');
	if drop:
		# Select users to drop based on the plotted histogram
		users_to_drop.update(data[data.weight > 200000].index)

	# Plot mean
	data = users.mean()
	data.hist(ax=ax2, column='weight', bins=30, log=True)
	ax2.set_title('Mean Weigth distribution over users')
	ax2.set_xlabel('weigth')
	if drop:
		# Select users to drop based on the plotted histogram
		users_to_drop.update(data[data.weight > 15000].index)

	# Plot number of different artists users listened to
	data = users.nunique().artistID
	data.hist(ax=ax3, bins=30, log=True);
	ax3.set_title('Different artists distribution over users')
	ax3.set_xlabel('Number of different artists')
	if drop:
		# Select users to drop based on the plotted histogram
		users_to_drop.update(data[data<10].index)

	if drop:
		print(len(users_to_drop), ' users to drop')
		return users_to_drop

def user_weight_distribution(normalized, seed=1):
	np.random.seed(seed)
	user = np.random.choice(normalized.userID)
	y = normalized[normalized.userID == user].weight
	plt.bar(np.arange(len(y)), y);
	plt.title('Distribution of weights for a random user');
	plt.xlabel('user');
	plt.ylabel('weight');

def degree_distribution(degree):
	degree_dict = dict(degree)
	data = list(degree_dict.values())

	plt.figure(figsize=(20,10))
	plt.hist(data,bins=120)

	plt.xticks(range(0, 120, 2))
	plt.title('Users\' degree distribution');
	plt.xlabel('degree');
	plt.ylabel('# users')
	plt.show()

def plot_tags_statistics(group):
	_, [ax1,ax2] = plt.subplots(1,2, figsize=(20,10))

	group.plot(use_index=False, ax=ax1, title='Tag distribution');
	ax1.set_xlabel('artistID')
	ax1.set_xlabel('# tags')

	# Tags distribution
	group.hist(ax=ax2, bins=50, log=True);
	ax2.set_xlabel('# tags');
	ax2.set_title('Tags Distribution');

def plot_separate_small_artist(small, big):
	_, [ax1,ax2] = plt.subplots(1, 2, figsize=(20,10));

	small.hist(ax=ax1, bins=30);
	ax1.set_xlabel('# tags');
	ax1.set_title('Histogram of artists with low tag frequency');

	big.hist(ax=ax2, bins=30, xlabelsize=8);
	ax2.set_xticks(np.round(np.linspace(min(big),max(big),10)))
	ax2.set_xlabel('# tags');
	ax2.set_title('Histogram of artists with high tag frequency');

def plot_unique_tags(group):
	group.plot(use_index=False);
	plt.title('Unique tags');
	plt.xlabel('artist ID');
	plt.ylabel('# tags')

def plot_listenig_count_frequency(max_user_weight):

	plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
	max_user_weight['weight'].plot.hist(bins=30)
	plt.xlabel('Listening count');
def plot_artist_per_user(number_user_artist):

	plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
	number_user_artist['artistID'].plot.hist(bins=10)
