#!/usr/bin/env python
# coding: utf-8

import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+' '+' '.join(map(str,args)).replace('\n',''))

# -------------------------------------------------------------------
import torchvision
def _load_dataset(data_name, split):
	transforms = [torchvision.transforms.ToTensor()]
	# normalize iputs
	if data_name=='MNIST':		transforms.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
	if data_name=='CIFAR10':	transforms.append(torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
	transforms = torchvision.transforms.Compose(transforms)
	# options are argument given to dataset
	options	= {'root':f'data/{data_name}','download':True,'transform':transforms, 'train':(split=='train')}
	# select dataclass
	if data_name=='MNIST':		dataclass = torchvision.datasets.MNIST
	if data_name=='CIFAR10':	dataclass = torchvision.datasets.CIFAR10
	return dataclass(**options)


import os
import torch
def _get_simplified_dataset(data_name, split, gpu):
	cache = f'data/.cache/{data_name}-{split}'
	if os.path.exists(cache):
		report(f'load cached xy in {data_name}-{split}')
		x = torch.load(f'{cache}/x.pt')
		y = torch.load(f'{cache}/y.pt')
	else:
		report(f'load dataset of {data_name}-{split}')
		dataset = _load_dataset(data_name, split)
		x = torch.stack( [v[0] for v in dataset])	# extract inputs
		y = torch.tensor([v[1] for v in dataset])	# extract labels
		report(f'caching dataset...')
		os.makedirs(cache, exist_ok=True)
		torch.save(x,f'{cache}/x.pt')
		torch.save(y,f'{cache}/y.pt')
	x = x.cuda(gpu)
	y = y.cuda(gpu)
	return x,y


# -------------------------------------------------------------------
import random, numpy, torch
def _set_random_seed(seed):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)	

# -------------------------------------------------------------------
import random, numpy, torch
class _SimplifiedGenerator(object):
	def __init__(self, x, y, batch_size, is_shuffle, augmentations=None):
		self.x, self.y	= x,y
		self.batch_size = batch_size
		self.is_shuffle = is_shuffle
		self.augmentations = augmentations
		self.n = len(x)
		self.seed		= None
		self.is_reverse	= False
	def __iter__(self):
		if not self.is_shuffle:	# for validation and test datasets
			for i in range(0, self.n, self.batch_size):
				yield self.x[i:i+self.batch_size], self.y[i:i+self.batch_size]
		else:				# for training dataset
			_set_random_seed(self.seed)
			self.index = torch.randperm(self.n)
			x = self.x[self.index]
			y = self.y[self.index]
			if self.augmentations is not None:
				x = self.augmentations(x).detach()
			if self.is_reverse: order = -1 # generate mini-batch in reverse order
			else:				order = +1
			for i in range(0, self.n, self.batch_size)[::order]:
				bx = x[i:i+self.batch_size]
				by = y[i:i+self.batch_size]
				bi = self.index[i:i+self.batch_size]
				yield bi,bx,by

# -------------------------------------------------------------------
import numpy
def _random_split(data, seed, size):
	x,y = data
	n = len(x)
	_set_random_seed(seed)
	index = numpy.random.permutation(n)
	train_x, train_y = x[index[size:]], y[index[size:]]
	valid_x, valid_y = x[index[:size]], y[index[:size]]	
	return (train_x, train_y), (valid_x, valid_y)

#from . import augmentation_zoo as aug_zoo
import augmentation_zoo as aug_zoo
from collections import namedtuple
def _generic_process(data_name, gpu, train_batch_size, test_batch_size, augmentations, seed, nval):
	train = _get_simplified_dataset(data_name=data_name, split='train', gpu=gpu)
	test  = _get_simplified_dataset(data_name=data_name, split='test',  gpu=gpu)
	# cut off validation dataset from train dataset to calculate query vector of LIEs
	report(f'set split-seed:{seed}')
	train, valid = _random_split(train,seed=seed,size=nval)
	augmentations = aug_zoo.get_transforms(augmentations)	# this argumentation aims to process data simultaneously
	train = _SimplifiedGenerator(*train, train_batch_size, is_shuffle=True, augmentations=augmentations)
	valid = _SimplifiedGenerator(*valid, test_batch_size,  is_shuffle=False)
	test  = _SimplifiedGenerator(*test,  test_batch_size,  is_shuffle=False)
	return namedtuple('_','train valid test')(train=train, valid=valid, test=test)	

#import torchvision
def MNIST(**kwargs):
#	kwargs['augmentations'] = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(28, padding=2, fill=-1)])	
	kwargs['augmentations'] = ['Crop2']
	return _generic_process(data_name='MNIST', 	  **kwargs)

def CIFAR10(**kwargs):
	kwargs['augmentations'] = ['Crop4','Flip']
	return _generic_process(data_name='CIFAR10',  **kwargs)

# -------------------------------------------------------------------




