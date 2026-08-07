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
def sgd_with_skip(model, datasets, skip_idx, config):
	report(f'\tsum-check:{sum(p.sum().item() for p in model.parameters())}')
	optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
	for epoch in range(config.start_epoch, config.end_epoch):
		# training
		model.train()
		datasets.train.seed = epoch
		x_sum = 0
		for i,x,y in datasets.train:
			x_sum += x.sum().item()

			# remove harmful example's index 
			wrong_mask = torch.tensor([iii in skip_idx for iii in i.numpy()])
			good_mask  = ~wrong_mask
			
			x = x[good_mask]
			y = y[good_mask]
			logit = model(x)
			loss  = F.cross_entropy(logit, y)
			
			optimizer.zero_grad()
			loss.backward()
			for param in model.parameters():
				param.grad.data *= sum(good_mask) / len(i)
			optimizer.step()
		report(f'\tepoch:{epoch} x_sum:{x_sum}')
	return model

import torch.nn.functional as F
def eval_infl(datasets, model, config):
	fn 			= f'{config.episode}/epoch{config.start_epoch}_model.dat'
	params		= torch.load(fn)
	init_param	= params[0]
	del params

	# infl
	infl_sgd_last	= joblib.load(f'{config.episode}/infl_sgd_at_epoch{config.target_epoch:02d}.dat')[:, -1]
	infl_sgd_all	= joblib.load(f'{config.episode}/infl_sgd_at_epoch{config.target_epoch:02d}.dat')[:, 0]

	np.random.seed(config.seed)
	infls = {'baseline':[],'sgd_last':infl_sgd_last, 'sgd_all':infl_sgd_all, 'random':np.random.rand(datasets.train.n)}
	
	k_list = [1, 3, 6, 10, 30, 60, 100, 300, 600, 1000, 3000, 6000, 10000]
		
	# eval
	score = {}
	for k in k_list:
		for key in infls.keys():
			if key in score: continue # prevent redundant calculation of baseline			
			skip_idx = np.argsort(infls[key])[:k]
			skip_idx = set(skip_idx)
			report(k, key)
			
			# sgd
			model.set_param(init_param)
			model = sgd_with_skip(model, datasets, skip_idx, config)

			loss_val, acc_val 	= commons.eval_model(model, datasets.valid)
			loss_te,  acc_te 	= commons.eval_model(model, datasets.test)

			if key == 'baseline':
				score[key]		= (loss_val, loss_te, acc_val, acc_te)
			else:
				score[key, k]	= (loss_val, loss_te, acc_val, acc_te)

	# save
	fn = f'{config.episode}/eval_epoch_{config.start_epoch:02d}_to_{config.target_epoch:02d}.dat'
	joblib.dump(score, fn, compress=9)
	report('finished')

# -------------------------------------------------------------------
import argparse, json
from collections import namedtuple
def get_config():
	args = argparse.ArgumentParser(add_help=False)
	args.add_argument('--episode',		default='workroom',	type=str)
	args.add_argument('--gpu',			default=0,			type=int)
	args.add_argument('--start_epoch',	default=19,			type=int)
	args.add_argument('--target_epoch',	default=20,			type=int)
	args.add_argument('--end_epoch',	default=20,			type=int)
	# load config of the given episode
	args = args.parse_args()
	with open(f'{args.episode}/config.json','r') as fp:
		config = json.load(fp)
	args = config | vars(args)	# carry over dataset, seed, nval, test_batch_size, bundle_size, lr, momentum, batch_size
	return namedtuple('_',args.keys())(**args)	# set all settings immutable

def main():
	config = get_config()	
	datasets	= commons.get_dataset(config)
	model		= commons.get_model(config)
	eval_infl(datasets, model, config)

if __name__ == '__main__':
	main()

