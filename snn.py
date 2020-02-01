import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from functional import LinearIF
import numpy as np
import copy

class sDropout(nn.Module):
	def __init__(self, layerType, pDrop):
		super(sDropout, self).__init__()

		self.pKeep = 1 - pDrop
		self.type = layerType # 1: Linear 2: Conv

	def forward(self, x_st, x_sc):
		if self.training:
			T = x_st.shape[1]
			mask = torch.bernoulli(x_sc.data.new(x_sc.data.size()).fill_(self.pKeep))/self.pKeep
			x_sc_out = x_sc * mask
			x_st_out = torch.zeros_like(x_st)
			
			for t in range(T):
				# Linear Layer
				if self.type == 1:
					x_st_out[:,t,:] = x_st[:,t,:] * mask
				# Conv1D Layer
				elif self.type == 2:
					x_st_out[:,t,:,:] = x_st[:,t,:,:] * mask
				# Conv2D Layer					
				elif self.type == 3:
					x_st_out[:,t,:,:,:] = x_st[:,t,:,:,:] * mask
		else:					
			x_sc_out = x_sc
			x_st_out = x_st
			
		return x_st_out, x_sc_out
		
class Linear(nn.Module):

	def __init__(self, D_in, D_out, net_params, device=torch.device('cuda'), bias=True):
		super(Linear, self).__init__()

		self.net_params = net_params
		self.linearif = LinearIF.apply
		self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
		self.device = device

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate linear layer
		T = input_feature_st.shape[1]
		output_round = torch.floor(self.linear(input_features_sc))
		output = torch.clamp(output_round, min=0, max=T)

		# extract the weight and bias from the surrogate linear layer
		linearif_weight = self.linear.weight.detach().to(self.device)
		linearif_bias = self.linear.bias.detach().to(self.device)

		# propagate the input spike train through the linearIF layer to get actual output
		# spike train
		output_st, output_sc = self.linearif(input_feature_st, output, linearif_weight, self.net_params, \
												self.device, linearif_bias)

		return output_st, output_sc

class LinearBN1d(nn.Module):

	def __init__(self, D_in, D_out, device=torch.device('cuda'), bias=True):
		super(LinearBN1d, self).__init__()
		self.linearif = LinearIF.apply
		self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
		self.device = device
		self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-4, momentum=0.9)
		nn.init.normal_(self.bn1d.weight, 0, 2.0)

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate linear layer
		T = input_feature_st.shape[1]
		output_bn = self.bn1d(self.linear(input_features_sc))
		output = F.relu(output_bn)

		# extract the weight and bias from the surrogate linear layer
		linearif_weight = self.linear.weight.detach().to(self.device)
		linearif_bias = self.linear.bias.detach().to(self.device)

		bnGamma = self.bn1d.weight
		bnBeta = self.bn1d.bias
		bnMean = self.bn1d.running_mean
		bnVar = self.bn1d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Linear' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0) 
		biasNorm = torch.mul(linearif_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the linearIF layer to get actual output
		# spike train
		output_st, output_sc = self.linearif(input_feature_st, output, weightNorm,  \
												self.device, biasNorm)

		return output_st, output_sc		





