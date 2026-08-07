#!/usr/bin/env python
# coding: utf-8

# these argumentations aim to process data simultaneously

# -------------------------------------------------------------------
import torch
class _All_RandomCrop:	# omit padding=None case
	def __init__(self, size, padding, fill=0):
		self.size	 	= size
		self.padding	= padding
		self.fill		= fill.reshape(1,len(fill),1,1)
		# assume "fill" matches channels in inputs
	def __call__(self, inputs):
		shape = list(inputs.shape)
		shape[2] += self.padding * 2
		shape[3] += self.padding * 2
		# 元々
#		padded = torch.zeros(shape, dtype=inputs.dtype, device=inputs.device)
#		padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = inputs
		# fillを組み込んだ版
		padded = torch.ones(shape, dtype=inputs.dtype, device=inputs.device)	# (b,c,w,h)
		fill = self.fill.to(inputs.device)
		padded = padded * fill
		padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = inputs
		
		w, h = padded.size(2), padded.size(3)
		th, tw = self.size, self.size
		i = torch.randint(0, h - th + 1, (inputs.size(0),), device=inputs.device)
		j = torch.randint(0, w - tw + 1, (inputs.size(0),), device=inputs.device)
		
		rows 	= torch.arange(th, dtype=torch.long, device=inputs.device) + i[:, None]
		columns = torch.arange(tw, dtype=torch.long, device=inputs.device) + j[:, None]
		padded = padded.permute(1, 0, 2, 3)
		
		b = torch.arange(inputs.size(0))[:, None, None]
		w = rows[:, torch.arange(th)[:, None]]
		h = columns[:, None]
		padded = padded[:,b,w,h]

		return padded.permute(1, 0, 2, 3)

# -------------------------------------------------------------------
import torch
class _All_RandomHorizontalFlip:
	def __init__(self, p=0.5, inplace=True):
		self.p = p
		self.inplace = inplace
	def __call__(self, inputs):
		if not self.inplace: inputs = inputs.clone()
		flipped = torch.rand(inputs.size(0)) < self.p
		inputs[flipped] = torch.flip(inputs[flipped], [3])
		return inputs

# -------------------------------------------------------------------
import torchvision
def get_transforms(augmentations):
	if len(augmentations)==0: return None
	transforms = []
	for keyword in augmentations:
		if keyword=='Crop2':
			bias	= torch.tensor([0.5])	# bias used to normalize in preprocess
			std		= torch.tensor([0.5])	# std  used to normalize in preprocess
			fill 	= -bias/std
			transforms.append(_All_RandomCrop(28, padding=2, fill=fill))		# alway create new variable. for MNIST
		if keyword=='Crop4':
			bias	= torch.tensor((0.4914, 0.4822, 0.4465))	# bias used to normalize in preprocess
			std		= torch.tensor((0.2023, 0.1994, 0.2010))	# std  used to normalize in preprocess
			fill 	= -bias/std
			transforms.append(_All_RandomCrop(32, padding=4, fill=fill))		# alway create new variable. for CIFAR-10/100
		if keyword=='Flip': 	transforms.append(_All_RandomHorizontalFlip())
	return torchvision.transforms.Compose(transforms)

	
# -------------------------------------------------------------------
