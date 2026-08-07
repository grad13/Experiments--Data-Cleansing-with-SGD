#!/usr/bin/env python
# coding: utf-8

import sys


# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))


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
	methods = ['sgd_last','sgd_all','random']

	ax = fig.add_subplot(1,1,1)

	y = buckets['baseline']
	ax.axhline(y=y,label='baseline', color='black',markersize=7,lw=1.)

	for i,m in enumerate(methods):
		y = []
		for k in Ks:
			y.append(buckets[(m,k)])
		ax.plot(Ks,y,label=m, marker='+',color=cm.tab10(i),markersize=7,lw=1.)
	ax.legend()
	ax.set_xscale('log')

	"""
#	if 'mnist' in file_name:
#		plt.ylim([0.007, 0.011])
#		yticks = [0.007, 0.008, 0.009, 0.01, 0.011]
#		ax.set_yticks(yticks)
	if 'cifar10' in file_name:
#		plt.ylim([0.155, 0.175])
		plt.ylim([0.15, 0.19])
		yticks = [0.15, 0.16, 0.17, 0.18, 0.19]
		ax.set_yticks(yticks)
	"""
			
	os.makedirs(os.path.dirname(file_name),exist_ok=True)
	plt.savefig(file_name,dpi=120)


# -------------------------------------------------------------------
import os
from collections import defaultdict
import joblib
def load_results(data):
	buckets = defaultdict(list)

	archive = f'MIRROR/{data}/'
	if not os.path.exists(archive): return
	for d in os.listdir(archive):
		if '.DS' in d: continue
		
		file_name = f'{archive}/{d}/eval_epoch_19_to_20.dat'
		if not os.path.exists(file_name): continue
		score = joblib.load(file_name)

		for k in score:
			test_error_rate = 1-score[k][-1]
			buckets[k].append(test_error_rate)
	buckets = dict(buckets)
	for k in buckets:
		buckets[k] = sum(buckets[k])/len(buckets[k])
	return buckets

# -------------------------------------------------------------------
def main():
	report('start draw')

	for data in ['mnist','cifar10']:
		buckets = load_results(data)
		if len(buckets)==0: continue
		output_file = f"RESULTs/{data}.png"
		draw(buckets, output_file)

	report('finish')
	print('-'*50)


# -------------------------------------------------------------------
if __name__=='__main__':
	main()


