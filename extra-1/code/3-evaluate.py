#!/usr/bin/env python
# coding: utf-8

import torch
import commons
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+' '+' '.join(map(str,args)).replace('\n',''))


# -------------------------------------------------------------------
import pickle, copy
import numpy as np
import torch.nn.functional as F
def run_counter_factually(datasets, model, config):
	init_param = torch.load(f'{config.episode}/params/{config.start_epoch:02d}.pt')
	init_param = init_param[0]	# select the init param in params

	def sgd_with_skip(removal_index):
		model.set_param(copy.deepcopy(init_param), gpu=config.gpu)
		optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
		for epoch in range(config.start_epoch, config.num_epochs):
			model.train()
			datasets.train.seed = epoch
			for i,x,y in datasets.train:
				wrong_mask = torch.tensor([iii.item() in removal_index for iii in i])
				good_mask  = ~wrong_mask
				good_mask  = good_mask.cuda(config.gpu)
				
				logit = model(x)
				loss  = F.cross_entropy(logit, y, reduction='none')
				loss  = (loss*good_mask).mean()
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
		measure = {}
		measure['valid'] = commons.eval_model(model, datasets.valid)
		measure['test']	 = commons.eval_model(model, datasets.test)
		return measure

	# load influence score
	file_name = f'{config.episode}/scores/{config.target_epoch:02d}.pkl'
	with open(file_name, 'rb') as fp:
		scores = pickle.load(fp)

	priority = {}
	priority['last']	= np.argsort(scores[-1,:])
	priority['all']		= np.argsort(scores.sum(0))
	np.random.seed(config.seed)
	priority['random']	= np.random.permutation(datasets.train.n)
		
	Ks = [1, 3, 6, 10, 30, 60, 100, 300, 600, 1000, 3000, 6000, 10000]
	
	evaluation = {}
	for method in priority:
		for k in Ks:
			report(f'method:{method} k:{k}')
			removal_index = set(priority[method][:k])
			evaluation[method, k] = sgd_with_skip(removal_index)
	report(f'method:baseline')
	removal_index = []
	evaluation['baseline'] = sgd_with_skip(removal_index)
	
	return evaluation


# -------------------------------------------------------------------
import argparse, json
from collections import namedtuple
def get_config():
	args = argparse.ArgumentParser(add_help=False)
	args.add_argument('--workroom',		default='workroom',	type=str)
	args.add_argument('--episode',		default='',			type=str)
	args.add_argument('--gpu',			default=0,			type=int)
	args.add_argument('--start_epoch',	default=19,			type=int)
	args.add_argument('--target_epoch',	default=20,			type=int)
	args = args.parse_args()
	with open(f'{args.episode}/config.json','r') as fp:
		config = json.load(fp)
	args = config | vars(args)	# carry over dataset, seed, nval, test_batch_size, lr, momentum, batch_size, num_epochs
	return namedtuple('_',args.keys())(**args)	# set all settings immutable

# -------------------------------------------------------------------
import os
import pickle
def main():
	print('-'*50)
	report('start')

	config 		= get_config()	
	datasets	= commons.get_dataset(config)
	model		= commons.get_model(config)
	evaluation	= run_counter_factually(datasets, model, config)


	os.makedirs(config.workroom, exist_ok=True)
	with open(f'{config.workroom}/config.json', 'w') as fp:
		json.dump(config._asdict(), fp, indent=2)

	os.makedirs(f'{config.episode}/evaluations/', exist_ok=True)
	file_name = f'{config.episode}/evaluations/{config.start_epoch:02d}_to_{config.target_epoch:02d}.pkl'
	with open(file_name, 'wb') as fp:
		pickle.dump(evaluation, fp)

	report('finished')
	print('-'*50)

if __name__ == '__main__':
	main()

