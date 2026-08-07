#!/usr/bin/env python
# coding: utf-8
import torch
torch.backends.cudnn.deterministic	= True
torch.backends.cudnn.benchmark		= False

# -------------------------------------------------------------------
import copy
import torch
class SaveLoadBase(torch.nn.Module):
	def get_param(self):
		state = copy.deepcopy(self.state_dict())
		for k in state:
			state[k] = state[k].cpu().detach()
		return state
	def set_param(self, param, gpu=None):
		if gpu is not None:
			param = {k:v.cuda(gpu) for k,v in param.items()}
		self.load_state_dict(param)

# -------------------------------------------------------------------
# just for debug
import torch
class SmallCNN(SaveLoadBase):
	def __init__(self, num_classes=10):
		super(SmallCNN, self).__init__()
		self.conv1 = torch.nn.Conv2d(1, 2, 5, stride=1, padding=0)
		self.L = torch.nn.Linear(2*5*5, num_classes)
	def forward(self, x):								# := (b,1,28,28)	# (batch, channel, width, height)
		x = torch.nn.functional.max_pool2d(x, 2, 2)		# -> (b,2,14,14)	
		x = torch.nn.functional.relu(self.conv1(x))		# -> (b,2,10,10)
		x = torch.nn.functional.max_pool2d(x, 2, 2)		# -> (b,2, 5, 5)
		x = x.flatten(1,-1)								# -> (b,2*5*5)
		return self.L(x)								# -> (b,10)

# -------------------------------------------------------------------
class MnistNet(SaveLoadBase):
	def __init__(self, num_classes=10):
		super(MnistNet, self).__init__()
		self.conv1 = torch.nn.Conv2d( 1, 20, 5, stride=1, padding=0)
		self.conv2 = torch.nn.Conv2d(20, 20, 5, stride=1, padding=0)
		self.L = torch.nn.Linear(4*4*20, num_classes)

	def forward(self, x):							# := (b,1,28,28)	# (batch, channel, width, height)
		x = torch.nn.functional.relu(self.conv1(x))	# -> (b,20,24,24)
		x = torch.nn.functional.max_pool2d(x, 2, 2)	# -> (b,20,12,12)
		x = torch.nn.functional.relu(self.conv2(x))	# -> (b,20, 8, 8)
		x = torch.nn.functional.max_pool2d(x, 2, 2)	# -> (b,20, 4, 4)
		x = x.flatten(1,-1)							# -> (b,20*4*4)
		return self.L(x)							# -> (b,10)

# -------------------------------------------------------------------	
import torch
class BasicBlock(torch.nn.Module):
	def __init__(self, dim1, dim2, dim3):
		super(BasicBlock, self).__init__()
		self.conv1	= torch.nn.Conv2d(dim1, dim2, kernel_size=3, padding=1)
		self.bn1	= torch.nn.BatchNorm2d(dim2)
		self.conv2	= torch.nn.Conv2d(dim2, dim3, kernel_size=3, padding=1)
	def forward(self, x):
		x = self.bn1(self.conv1(x))
		x = torch.nn.functional.relu(x)
		x = self.conv2(x)
		x = torch.nn.functional.max_pool2d(x, 2, 2)
		x = torch.nn.functional.relu(x)
		return x

class CifarNet(SaveLoadBase):
	def __init__(self, num_classes=10):
		super(CifarNet, self).__init__()
		self.block1 = BasicBlock( 3, 32, 32)
		self.block2 = BasicBlock(32, 64, 64)
		self.block3 = BasicBlock(64,128,128)
		self.L		= torch.nn.Linear(16*128, num_classes)
	
	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = x.flatten(1, -1)
		return self.L(x)

