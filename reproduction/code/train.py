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
def train(datasets, model, config):
	optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
					
	# training
	score = []
	torch.manual_seed(config.seed)
	for epoch in range(config.num_epochs):
		# training
		model.train()
		datasets.train.seed = epoch
		
		params = []
		loss_tr, acc_tr = 0,0
		x_sum = 0
		for _,x,y in datasets.train:
			params.append(model.get_param())

			logit = model(x)
			loss  = F.cross_entropy(logit, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			loss_tr	+= loss.item() * len(x)
			acc_tr	+= (logit.argmax(dim=1)==y).sum().item()

			x_sum += x.sum().item()
		params.append(model.get_param())

		loss_tr /= datasets.train.n
		acc_tr  /= datasets.train.n
			
		torch.save(params,					 f'{config.output_path}/epoch{epoch:02d}_model.dat')
		torch.save(optimizer.state_dict(),	f'{config.output_path}/epoch{epoch:02d}_optimizer.dat')
		
		# evaluation
		loss_val, acc_val	= commons.eval_model(model, datasets.valid)
#		loss_tes, acc_tes	= commons.eval_model(model, datasets.test)
				
		message = []
		message.append(f'epoch:{epoch}/{config.num_epochs}')
		message.append(f'\ttrain loss:{loss_tr: 15.7f} accuracy:{acc_tr: 15.7f}')
		message.append(f'\tvalid loss:{loss_val: 15.7f} accuracy:{acc_val: 15.7f}')
#		message.append(f'\ttest  loss:{loss_tes: 15.7f} accuracy:{acc_tes: 15.7f}')
		message.append(f'\tsum-check param:{sum(p.sum().item() for p in model.parameters())}')
		message.append(f'\tsum-check input:{x_sum}')
		for m in message:
			report(m)

#		score.append((loss_tr, loss_val, loss_tes, acc_tr, acc_val, acc_tes))
		score.append((loss_tr, loss_val, acc_tr, acc_val))
		
	# save
	fn = f'{config.output_path}/score.dat'
	joblib.dump(np.array(score), fn, compress=9)
	report('finished')


# -------------------------------------------------------------------
import argparse
from collections import namedtuple
def get_config():
	args = argparse.ArgumentParser(add_help=False)
	args.add_argument('--output_path',		default='workroom',	type=str)
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

import json
def main():
	config      = get_config()
	
	datasets	= commons.get_dataset(config)
	model		= commons.get_model(config)

	os.makedirs(config.output_path, exist_ok=True)
	train(datasets, model, config)

	with open(f'{config.output_path}/config.json', 'w') as fp:
		json.dump(config._asdict(), fp, indent=2)

	
if __name__ == '__main__':
	main()
