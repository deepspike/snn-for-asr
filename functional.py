import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import numpy as np
import copy

class ZeroExpandInput(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input_image, T, device=torch.device('cuda')):
		"""
		Args:
			input_image: normalized within (0,1)
		"""
		N, dim = input_image.shape
		input_image_sc = input_image
		zero_inputs = torch.zeros(N, T-1, dim).to(device)
		input_image = input_image.unsqueeze(dim=1)
		input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

		return input_image_spike, input_image_sc

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		return None, None, None

class LinearIF(torch.autograd.Function):

	@staticmethod
	def forward(ctx, spike_in, ann_output, weight, device=torch.device('cuda'), bias=None):
		"""
		args:
			spike_in: (N, T, in_features)
			weight: (out_features, in_features)
			bias: (out_features)
		"""
		N, T, _ = spike_in.shape
		out_features = bias.shape[0]
		pot_in = spike_in.matmul(weight.t())
		spike_out = torch.zeros_like(pot_in, device=device)
		pot_aggregate = bias.repeat(N, 1) # init the membrane potential with the bias

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			pot_aggregate += pot_in[:,t,:].squeeze()
			bool_spike = torch.ge(pot_aggregate, 1.0).float()
			spike_out[:,t,:] = bool_spike
			pot_aggregate -= bool_spike 

		spike_count_out = torch.sum(spike_out, dim=1).squeeze()

		return spike_out, spike_count_out

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""
		grad_ann_out = grad_spike_count_out.clone()

		return None, grad_ann_out, None, None, None, None
