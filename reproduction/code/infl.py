import os, sys
import copy
import numpy as np
import joblib
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
def infl_sgd(datasets, model, config):
	# model setup
	fn = f'{config.episode}/epoch{config.target_epoch-1:02d}_model.dat'
	params = torch.load(fn)
	model.set_param(params[-1], gpu=config.gpu)
#	exgrads.hooks.register(model)
			
	# gradient
	u = commons.compute_gradient(model, datasets.valid)
	report('query')
	report(f'target-epoch:{config.target_epoch}')
	report(f'model sum-check:{sum(p.sum().item() for p in model.parameters())}')
	report(f'query sum-check:{sum(v.sum() for v in u)}')
		
	infl = torch.zeros(datasets.train.n, config.target_epoch, requires_grad=False).cuda(config.gpu)
	for epoch in range(config.target_epoch-1, -1, -1):
		fn = f'{config.episode}/epoch{epoch:02d}_model.dat'
		params = torch.load(fn)
		params = params[:-1]

		datasets.train.is_reverse	= True
		datasets.train.seed 		= epoch

		x_sum = 0
		for (i,x,y),p in zip(datasets.train, params[::-1]):
			x_sum += x.sum().item()
			
			model.set_param(p, gpu=config.gpu)
						
			# calculate example-wise influence score
			model.eval()
			exgrads.hooks.register(model)
			logit = model(x)
			loss = F.cross_entropy(logit, y)
			loss.backward()
			exgrads.hooks.compute_grad1(model)
			
			score = 0
			for uu,p in zip(u,model.parameters()):
				score += config.lr * (uu * p.grad1).flatten(1,-1).sum(1)
			for iii,s in zip(i,score):
				infl[iii, epoch] += s
			exgrads.hooks.deregister(model)
			
			# update query vector
			model.train()
			logit = model(x)
			loss = F.cross_entropy(logit, y)
			grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
			ug = 0
			for uu, g in zip(u, grad_params):
				ug += (uu * g).sum()
			model.zero_grad()
			ug.backward()
			for j, param in enumerate(model.parameters()):
				u[j] -= config.lr * param.grad.data / len(i)

		if epoch > 0:
			infl[:, epoch-1] = infl[:, epoch].clone()		
		report(f'score:{epoch} sum-check:{sum(infl[:, epoch])} x-sum:{x_sum}')
				
	# save
	fn = f'{config.episode}/infl_sgd_at_epoch{config.target_epoch:02d}.dat'
	joblib.dump(infl.cpu().numpy(), fn, compress=9)
	report('finished')


# -------------------------------------------------------------------
import argparse, json
from collections import namedtuple
def get_config():
	args = argparse.ArgumentParser(add_help=False)
	args.add_argument('--episode',		default='workroom',	type=str)
	args.add_argument('--gpu',			default=0,			type=int)
	args.add_argument('--target_epoch',	default=20,			type=int)
	args = args.parse_args()
	with open(f'{args.episode}/config.json','r') as fp:
		config = json.load(fp)
	args = config | vars(args)	# dataset, seed, nval, test_batch_size, bundle_size
	return namedtuple('_',args.keys())(**args)	# set all settings immutable

def main():
	config = get_config()

	datasets	= commons.get_dataset(config)
	model		= commons.get_model(config)

	infl_sgd(datasets, model, config)


if __name__ == '__main__':
	main()
