 #!/usr/bin/env python
# coding: utf-8

# pip install -r experiment/requirements.txt

# ------------------------------------------------------
import os
from os import path
import query_queue_and_parallel as qqp
def main():
	available_gpu	= list(range(8))
	num_of_trial	= 8

	# train
	query = qqp.Query(archive='ARCHIVEs/TRAINs')
	query['scripts']		= [path.join(path.dirname(__file__),'code/1-train.py')]
	query['dataset']		= ['mnist','cifar10']
	query['seed']			= list(range(num_of_trial))
	query['num_epochs']		= [20]
	query.ready()
	query.run(GPU_index=available_gpu)

	archive = 'ARCHIVEs/TRAINs'
	episodes = [f'{archive}/{e}' for e in os.listdir(archive)]

	# score
	query = qqp.Query(archive='ARCHIVEs/SCOREs')
	query['scripts']		= [path.join(path.dirname(__file__),'code/2-score.py')]
	query['episode']		= episodes
	query['target_epoch']	= [20]
	query.ready()
	query.run(GPU_index=available_gpu)
	
	# run counter-factually
	query = qqp.Query(archive='ARCHIVEs/EVALUATIONs')
	query['scripts']		= [path.join(path.dirname(__file__),'code/3-evaluate.py')]
	query['episode']		= episodes
	query['start_epoch']	= [0,19]
	query['target_epoch']	= [20]
	query.ready()
	query.run(GPU_index=available_gpu)
	
if __name__ == '__main__':
	main()



# ----------------------
