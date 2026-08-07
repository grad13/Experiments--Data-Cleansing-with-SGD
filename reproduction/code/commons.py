#!/usr/bin/env python
# coding: utf-8

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+' '+' '.join(map(str,args)).replace('\n',''))

# -------------------------------------------------------------------
import torch.nn.functional as F
def compute_gradient(model, dataset):
	model.eval()
	# calculate query vector using validation dataset
	model.zero_grad()
	for x,y in dataset:
		logit = model(x)
		loss  = F.cross_entropy(logit, y, reduction='sum')
		loss.backward()
	# chone parameters and scale them by 1/n
	query = [p.grad.data.clone()/dataset.n for p in model.parameters()]
	query = [q.requires_grad_(False) for q in query]
	return query

# -------------------------------------------------------------------
import torch.nn.functional as F
def eval_model(model, dataset):
	model.eval()
	with torch.no_grad():
		Loss, accuracy = 0,0
		for x,y in dataset:
			logit	= model(x)
			loss	= F.cross_entropy(logit, y, reduction='sum')
			Loss 		+= loss.item()
			accuracy	+= (logit.argmax(dim=1)==y).sum().item()
	return Loss/dataset.n, accuracy/dataset.n

# -------------------------------------------------------------------
#from . import data_zoo
import data_zoo
def get_dataset(config):
	args = {}
	args['train_batch_size'] = config.batch_size
	args['test_batch_size']  = config.test_batch_size
	args['seed'] 			 = config.seed
	args['gpu']				 = config.gpu
	args['nval']			 = config.nval
	if config.dataset=='mnist':
		return data_zoo.MNIST(**args)
	if config.dataset=='cifar10':
		return data_zoo.CIFAR10(**args)

#from . import model_zoo
import model_zoo
def get_model(config):
	torch.manual_seed(config.seed)
	if config.dataset=='mnist':
		model = model_zoo.MnistNet()
	if config.dataset=='cifar10':
		model = model_zoo.CifarNet()
	model = model.cuda(config.gpu)
	return model




# -------------------------------------------------------------------

