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
import os
import torch.nn.functional as F
def train(datasets, model, config):
	optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
	
	def update(dataset, epoch):
		measure = {'loss':0, 'accuracy':0, 'x-sum':0}
		params  = [model.get_param()]
		model.train()
		datasets.train.seed = epoch
		for _,x,y in datasets.train:
			logit = model(x)
			loss  = F.cross_entropy(logit, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			measure['loss']		+= loss.item() * len(x)
			measure['accuracy']	+= (logit.argmax(dim=1)==y).sum().item()
			measure['x-sum']	+= x.sum().item()
			params.append(model.get_param())
		os.makedirs(f'{config.workroom}/params', exist_ok=True)
		torch.save(params, f'{config.workroom}/params/{epoch:02d}.pt')
		return {k:v/dataset.n for k,v in measure.items()}

	trails = []
	for epoch in range(config.num_epochs):
		status = {}
		status['train']	= update(datasets.train, epoch)
		status['valid']	= commons.eval_model( model, datasets.valid)
		status['test']	= commons.eval_model( model, datasets.test)
		trails.append(status)
		
		report(f'epoch:{epoch}/{config.num_epochs}')
		for key in ['train','valid','test']:
			message	 = [f'\t{key:6}']
			message	+= [f"loss:{status[key]['loss']: 15.7f}"]
			message	+= [f"accuracy:{status[key]['accuracy']: 9.7f}"]
			report(*message)
		report(f'\tsum-check param:{sum(p.sum().item() for p in model.parameters())}')
		report(f'\tsum-check input:{status["train"]["x-sum"]}')
	return trails	

# -------------------------------------------------------------------
import argparse
from collections import namedtuple
def get_config():
	args = argparse.ArgumentParser(add_help=False)
	args.add_argument('--workroom',			default='workroom',	type=str)
	args.add_argument('--dataset',			default='mnist',	type=str,	choices=['mnist','cifar10'])
	args.add_argument('--seed',				default=0,			type=int)
	args.add_argument('--gpu',				default=0,			type=int)
	args.add_argument('--nval',				default=10000,		type=int)		# for data
	args.add_argument('--batch_size',		default=64,			type=int)		# for training
	args.add_argument('--lr',				default=0.05,		type=float)
	args.add_argument('--momentum',			default=0.0,		type=float)
	args.add_argument('--num_epochs',		default=3,			type=int)
	args.add_argument('--test_batch_size',	default=1000,		type=int)	# for evaluation
	args.add_argument('--bundle_size',		default=200,		type=int)	# for storing
	args = vars(args.parse_args())
	return namedtuple('_',args.keys())(**args)	# set all settings immutable

# -------------------------------------------------------------------
import commons
import json, pickle,os
def main():
	print('-'*50)
	report('start')

	config      = get_config()
	datasets	= commons.get_dataset(config)
	model		= commons.get_model(config)
	trails 		= train(datasets, model, config)

	os.makedirs(config.workroom, exist_ok=True)
	with open(f'{config.workroom}/config.json', 'w') as fp:
		json.dump(config._asdict(), fp, indent=2)
	with open(f'{config.workroom}/trails.pkl', 'wb') as fp:
		pickle.dump(trails, fp)

	report('finished')
	print('-'*50)
	
if __name__ == '__main__':
	main()
