#!/usr/bin/env python
# coding: utf-8

import sys


# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib import pyplot as plt

import numpy
def draw(buckets, file_name):	
	print(f'draw:{file_name}')
	fig = plt.figure(figsize=(5,3.5))
	plt.subplots_adjust(left=0.17, right=0.99, top=0.92, bottom=0.15)

	Ks = [1, 3, 6, 10, 30, 60, 100, 300, 600, 1000, 3000, 6000, 10000]
	methods = ['last','all','random']

	ax = fig.add_subplot(1,1,1)
	y = buckets['baseline']
	ax.axhline(y=y,label='baseline', color='black', lw=1.0, zorder=1)

	for i,m in enumerate(methods):
		x,y = [],[]
		for k in Ks:
			x.append(k)
			y.append(buckets[(m,k)])
		if i==0: label = 'using only last score'
		if i==1: label = 'using all epoch score'
		if i==2: label = 'random removal'
		ax.plot(x,y,label=label, marker='+', color=cm.tab10(i), markersize=7, lw=1.0, zorder=2)
	legend = ax.legend(loc='upper left')
	legend.get_frame().set_alpha(1)
	ax.set_xscale('log')

	ax.set_ylabel('miss-classification rate')
	ax.set_xlabel('number of removing instance')

	if 'mnist' in file_name:
		plt.ylim([0.007, 0.011])
		yticks = [0.007, 0.008, 0.009, 0.01, 0.011]
		ax.set_yticks(yticks)
		for y in yticks:
			ax.axhline(y=y, color='black', lw=0.3, linestyle='--', zorder=-1, alpha=0.5)
	if 'cifar10' in file_name:
		plt.ylim([0.14, 0.17])
		yticks = [0.14, 0.15, 0.16, 0.17, 0.18]
		ax.set_yticks(yticks)
		for y in yticks:
			ax.axhline(y=y, color='black', lw=0.3, linestyle='--', zorder=-1, alpha=0.5)
	xticks = [1, 10, 100, 1000, 10000]
	for x in xticks:
		ax.axvline(x=x, color='black', lw=0.3, linestyle='--', zorder=-1, alpha=0.5)
		
	os.makedirs(os.path.dirname(file_name),exist_ok=True)
	plt.savefig(file_name,dpi=120)


# -------------------------------------------------------------------
import os, json, pickle
from collections import defaultdict
def load_results(data, start_epoch):
	buckets = defaultdict(list)

	archive = 'MIRROR/ARCHIVEs/TRAINs/'
	if not os.path.exists(archive): return
	for d in os.listdir(archive):
		if '.DS' in d: continue

		file_name = f'{archive}/{d}/config.json'
		with open(file_name, 'r') as fp:
			config = json.load(fp)
		if config['dataset']!=data: continue

		file_name = f'{archive}/{d}/evaluations/{start_epoch:02d}_to_20.pkl'
		if not os.path.exists(file_name): continue
		with open(file_name, 'rb') as fp:
			score = pickle.load(fp)

		for k in score:
			test_error_rate = 1-score[k]['test']['accuracy']
			buckets[k].append(test_error_rate)
	if len(buckets)==0: return

	buckets = dict(buckets)
	for k in buckets:
		buckets[k] = sum(buckets[k])/len(buckets[k])
	return buckets

# -------------------------------------------------------------------
def main():
	report('start draw')	

	for data in ['mnist','cifar10']:
		for start_epoch in [0,19]:
			buckets = load_results(data, start_epoch)
			if buckets is None: continue
			output_file = f"VIEW/data={data}-start_epoch={start_epoch:02d}.png"
			draw(buckets, output_file)

	report('finish')
	print('-'*50)


# -------------------------------------------------------------------
if __name__=='__main__':
	main()


