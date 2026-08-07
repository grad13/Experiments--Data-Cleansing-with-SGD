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
def draw(scores, file_name):
	print(f'draw:{file_name}')
	fig = plt.figure(figsize=(5,3.5))
	plt.subplots_adjust(left=0.17, right=0.99, top=0.92, bottom=0.15)

	ax = fig.add_subplot(1,1,1)

	for i,m in enumerate(methods):
		x,y = [],[]
		for k in Ks:
			x.append(k+1e-1)
			y.append(buckets[(m,k)])
		ax.plot(x,y,label=m, marker='+',color=cm.tab10(i),markersize=7,lw=1.)
	ax.legend()
	ax.set_xscale('log')
				
	os.makedirs(os.path.dirname(file_name),exist_ok=True)
	plt.savefig(file_name,dpi=120)


# -------------------------------------------------------------------
import os
from collections import defaultdict
import joblib
def main():
	report('start draw')

	for data in ['mnist','cifar10']:
		archive = f'MIRROR/{data}/'
		if not os.path.exists(archive): continue
		for d in os.listdir(archive):
			if '.DS' in d: continue

			file_name = f'{archive}/{d}/infl_sgd_at_epoch20.dat'
			if not os.path.exists(file_name): continue

			scores = joblib.load(file_name)
			# infl[:, epoch-1] = infl[:, epoch].clone()
			for i in range(20):
				pass

			output_file = f"VIEWs/{data}-{d}.png"
			draw(scores, output_file)

	report('finish')
	print('-'*50)


# -------------------------------------------------------------------
if __name__=='__main__':
	main()


