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
import torch.nn.functional as F
import exgrads
def get_scores(datasets, model, config):
	# calculate query vector
	params = torch.load(f'{config.episode}/params/{config.target_epoch-1:02d}.pt')
	model.set_param(params[-1], gpu=config.gpu)		
	query = commons.compute_gradient(model, datasets.valid)
	report('query')
	report(f'target-epoch:{config.target_epoch}')
	report(f'model sum-check:{sum(p.sum().item() for p in model.parameters())}')
	report(f'query sum-check:{sum(v.sum() for v in query)}')

	def update(dataset ,epoch):
		measure = {'score':torch.zeros(dataset.n).cuda(config.gpu), 'x-sum':0}
		params = torch.load(f'{config.episode}/params/{epoch:02d}.pt')
		params = params[:-1] # remove a redundant param
		dataset.is_reverse	= True
		dataset.seed 		= epoch
		for i,x,y in dataset:
			measure['x-sum'] += x.sum().item()
			model.set_param(params.pop(), gpu=config.gpu)
			
			# calculate example-wise influence score
			model.train()
			for m in model.modules(): # run batch-norm but not update running stats
				if hasattr(m, 'track_running_stats'):
					m.track_running_stats = False

			exgrads.hooks.register(model)
			logit = model(x)
			loss = F.cross_entropy(logit, y)
			loss.backward()
			exgrads.hooks.compute_grad1(model)

			for q,p in zip(query,model.parameters()):
				measure['score'][i] += config.lr * (q * p.grad1).flatten(1,-1).sum(1)
			exgrads.hooks.deregister(model)
			
			# update query vector
			model.train()
			logit = model(x)
			loss  = F.cross_entropy(logit, y)
			grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
			qg = sum( (q*g).sum() for q,g in zip(query, grads) )
			model.zero_grad()
			qg.backward()
			for j, param in enumerate(model.parameters()):
				query[j] -= config.lr * param.grad.data
		measure['score']  = measure['score'].cpu()
		measure['x-sum'] /= dataset.n
		return measure

	scores = [None for _ in range(config.target_epoch)]
	for epoch in range(config.target_epoch-1, -1, -1):
		measure = update(datasets.train, epoch)
		scores[epoch] = measure['score']

		report(f'score:{epoch}')
		report(f'\tscore sum-check:{sum(measure["score"])}')
		report(f'\tinput sum-check:{measure["x-sum"]}')
	scores = torch.stack(scores)
	return scores.numpy()
	
# -------------------------------------------------------------------
import argparse, json
from collections import namedtuple
def get_config():
	args = argparse.ArgumentParser(add_help=False)
	args.add_argument('--workroom',		default='workroom',	type=str)
	args.add_argument('--episode',		default='',			type=str)
	args.add_argument('--gpu',			default=0,			type=int)
	args.add_argument('--target_epoch',	default=20,			type=int)
	args = args.parse_args()
	with open(f'{args.episode}/config.json','r') as fp:
		config = json.load(fp)
	args = config | vars(args)	# dataset, seed, nval, test_batch_size
	return namedtuple('_',args.keys())(**args)	# set all settings immutable

# -------------------------------------------------------------------
import pickle, os
def main():
	print('-'*50)
	report('start')

	config 		= get_config()
	datasets	= commons.get_dataset(config)
	model		= commons.get_model(config)
	scores		= get_scores(datasets, model, config)

	os.makedirs(config.workroom, exist_ok=True)
	with open(f'{config.workroom}/config.json', 'w') as fp:
		json.dump(config._asdict(), fp, indent=2)
	
	os.makedirs(f'{config.episode}/scores/', exist_ok=True)
	with open(f'{config.episode}/scores/{config.target_epoch:02d}.pkl', 'wb') as fp:
		pickle.dump(scores, fp)

	report('finished')
	print('-'*50)


if __name__ == '__main__':
	main()
