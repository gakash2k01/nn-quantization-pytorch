import os, sys, time, random
proj_root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(proj_root_dir)
import argparse
import torch
import torchvision.models as models
import scipy.optimize as opt
from pathlib import Path
import numpy as np
import torch.nn as nn
from itertools import count
import torch.backends.cudnn as cudnn
from quantization.quantizer import ModelQuantizer
import pickle
from tqdm import tqdm

from utils.data import get_dataset
from utils.preprocess import get_transform
from utils.meters import AverageMeter, ProgressMeter, accuracy
from models.resnet import resnet as custom_resnet
from models.inception import inception_v3 as custom_inception
from utils.misc import normalize_module_name, arch2depth
from utils.absorb_bn import search_absorbe_bn

# uniform.py starts
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class QuantizationBase(object):
    def __init__(self, module, num_bits):
        self.module = module
        self.num_bits = num_bits
        self.num_bins = int(2 ** num_bits)
        self.opt_params = {}
        self.named_params = []

    def register_buffer(self, name, value):
        if hasattr(self.module, name):
            delattr(self.module, name)
        self.module.register_buffer(name, value)
        setattr(self, name, getattr(self.module, name))

    def register_parameter(self, name, value):
        if hasattr(self.module, name):
            delattr(self.module, name)
        self.module.register_parameter(name, nn.Parameter(value))
        setattr(self, name, getattr(self.module, name))

        self.named_params.append((name, getattr(self.module, name)))

    def __add_optim_params__(self, optim_type, dataset, params):
        learnable_params = [d for n, d in params if n in self.learned_parameters()]
        self.opt_params[optim_type + '_' + dataset] = learnable_params

    def optim_parameters(self):
        return self.opt_params

    def loggable_parameters(self):
        return self.named_parameters()

    def named_parameters(self):
        named_params = [(n, p) for n, p in self.named_params if n in self.learned_parameters()]
        return named_params

    @staticmethod
    def learned_parameters():
        return []


class UniformQuantization(QuantizationBase):
    def __init__(self, module, num_bits, symmetric, uint=False, stochastic=False, tails=False):
        super(UniformQuantization, self).__init__(module, num_bits)
        if not symmetric and not uint:
            raise RuntimeError("Can't perform integer quantization on non symmetric distributions.")

        self.symmetric = symmetric
        self.uint = uint
        self.stochastic = stochastic
        self.tails = tails
        if uint:
            self.qmax = 2 ** self.num_bits - 1
            self.qmin = 0
        else:
            self.qmax = 2 ** (self.num_bits - 1) - 1
            self.qmin = -self.qmax - 1

        if tails:
            self.qmax -= 0.5 + 1e-6
            self.qmin -= 0.5

    def __quantize__(self, tensor, alpha):
        delta = (2 if self.symmetric else 1) * alpha / (self.num_bins - 1)
        delta = max(delta, 1e-8)

        # quantize
        if self.uint and self.symmetric:
            t_q = (tensor + alpha) / delta
        else:
            t_q = tensor / delta

        # stochastic rounding
        if self.stochastic and self.module.training:
            with torch.no_grad():
                noise = t_q.new_empty(t_q.shape).uniform_(-0.5, 0.5)
                t_q += noise

        # clamp and round
        t_q = torch.clamp(t_q, self.qmin, self.qmax)
        t_q = RoundSTE.apply(t_q)
        assert torch.unique(t_q).shape[0] <= self.num_bins

        # uncomment to debug quantization
        # print(torch.unique(t_q))

        # de-quantize
        if self.uint and self.symmetric:
            t_q = t_q * delta - alpha
        else:
            t_q = t_q * delta

        return t_q

    # def __distiller_quantize__(self, tensor, alpha):
    #     # Leave one bit for sign
    #     n = self.qmax
    #     scale = n / alpha
    #     t_q = torch.clamp(torch.round(tensor * scale), self.qmin, self.qmax)
    #     t_q = t_q / scale
    #     return t_q

    def __quantize_gemmlowp__(self, tensor, min_, max_):
        assert self.uint is True
        delta = (max_ - min_) / (self.num_bins - 1)
        delta = max(delta, 1e-8)

        # quantize
        t_q = (tensor - min_) / delta

        # stochastic rounding
        if self.stochastic and self.module.training:
            with torch.no_grad():
                noise = t_q.new_empty(t_q.shape).uniform_(-0.5, 0.5)
                t_q += noise

        # clamp and round
        t_q = torch.clamp(t_q, self.qmin, self.qmax)
        t_q = RoundSTE.apply(t_q)
        assert torch.unique(t_q).shape[0] <= self.num_bins

        # uncomment to debug quantization
        # print(torch.unique(t_q))

        # de-quantize
        t_q = t_q * delta + min_

        return t_q

    def __for_repr__(self):
        return [('bits', self.num_bits), ('symmetric', self.symmetric), ('tails', self.tails)]

    def __repr__(self):
        s = '{} - ['.format(type(self).__name__)
        for name, value in self.__for_repr__():
            s += '{}: {}, '.format(name, value)
        return s + ']'
        # return '{} - bits: {}, symmetric: {}'.format(type(self).__name__, self.num_bits, self.symmetric)


class MaxAbsDynamicQuantization(UniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, stochastic=False):
        super(MaxAbsDynamicQuantization, self).__init__(module, tensor, num_bits, symmetric)

    def __call__(self, tensor):
        alpha = tensor.abs().max()
        t_q = self.__quantize__(tensor, alpha)
        return t_q


class MinMaxQuantization(UniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(MinMaxQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            self.register_buffer('min', tensor.new_tensor([tensor.min()]))
            self.register_buffer('max', tensor.new_tensor([tensor.max()]))

    def __call__(self, tensor):
        t_q = self.__quantize_gemmlowp__(tensor, min_=self.min, max_=self.max)
        return t_q
# uniform.py ends
# kmeans.py starts
def forgy(X, n_clusters):
	_len = len(X)
	indices = np.random.choice(_len, n_clusters)
	initial_state = X[indices]
	return initial_state


def lloyd1d(X, n_clusters, tol=1e-4, device=None, max_iter=100, init_state=None):
	if device is not None:
		X = X.to(device)

	if init_state is None:
		initial_state = forgy(X, n_clusters).flatten()
	else:
		initial_state = init_state.clone()

	iter = 0
	dis = X.new_empty((n_clusters, X.numel()))
	choice_cluster = X.new_empty(X.numel()).int()
	centers = torch.arange(n_clusters, device=X.device).view(-1, 1).int()
	initial_state_pre = initial_state.clone()
	# temp = X.new_empty((n_clusters, X.numel()))
	while iter < max_iter:
		iter += 1

		# Calculate pair wise distance
		dis[:, ] = X.view(1, -1)
		dis.sub_(initial_state.view(-1, 1))
		dis.pow_(2)

		choice_cluster[:] = torch.argmin(dis, dim=0).int()

		initial_state_pre[:] = initial_state

		temp = X.view(1, -1) * (choice_cluster == centers).float()
		initial_state[:] = temp.sum(1) / (temp != 0).sum(1).float()

		# center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
		center_shift = torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2))

		if center_shift < tol:
			break

	return choice_cluster, initial_state

# kmeans.py ends


# pairwise.py starts
'''
calculation of pairwise distance, and return condensed result, i.e. we omit the diagonal and duplicate entries and store everything in a one-dimensional array
'''
def pairwise_distance(data1, data2=None, device=-1):
	r'''
	using broadcast mechanism to calculate pairwise ecludian distance of data
	the input data is N*M matrix, where M is the dimension
	we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
	then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
	'''
	if data2 is None:
		data2 = data1

	if device!=-1:
		data1, data2 = data1.cuda(device), data2.cuda(device)

	#N*1*M
	A = data1.unsqueeze(dim=1)

	#1*N*M
	B = data2.unsqueeze(dim=0)

	dis = (A-B)**2.0
	#return N*N matrix for pairwise distance
	dis = dis.sum(dim=-1).squeeze()
	return dis

def group_pairwise(X, groups, device=0, fun=lambda r,c: pairwise_distance(r, c).cpu()):
	group_dict = {}
	for group_index_r, group_r in enumerate(groups):
		for group_index_c, group_c in enumerate(groups):
			R, C = X[group_r], X[group_c]
			if device!=-1:
				R = R.cuda(device)
				C = C.cuda(device)
			group_dict[(group_index_r, group_index_c)] = fun(R, C)
	return group_dict
# pairwise.py ends


# non-uniform.py starts
class ArgmaxMaskSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        output = input.new_full(input.shape, 0., requires_grad=True)
        output[torch.arange(output.shape[0]), input.argmax(dim=dim)] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class StepQuantizationSte(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, centroids):
        b = (centroids[1:] + centroids[:-1]) / 2
        i = (tensor.view(-1, 1) > b).sum(1)
        return centroids[i].view(tensor.shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class KmeansQuantization(object):
    def __init__(self, num_bits, max_iter=100, rho=0.5, uniform_init=False):
        self.num_bits = num_bits
        self.num_bins = int(2 ** num_bits)
        self.max_iter = max_iter
        self.rho =rho
        self.uniform_init = uniform_init

    def clustering(self, tensor):
        if self.uniform_init:
            # Initialize k-means centroids uniformaly
            rho = 0.5
            bin_size = (tensor.max() - tensor.min()) / self.num_bins
            zp = torch.round(tensor.min() / bin_size)
            x = (torch.arange(self.num_bins, device=tensor.device, dtype=tensor.dtype) + zp) * bin_size
            init = rho * torch.where(x != 0, x + bin_size / 2, x)
        else:
            init = None

        with torch.no_grad():
            cluster_idx, centers = lloyd1d(tensor.flatten(), self.num_bins, max_iter=self.max_iter, init_state=init, tol=1e-5)

        # workaround for out of memory issue
        torch.cuda.empty_cache()

        B = torch.zeros(tensor.numel(), self.num_bins, device=tensor.device)
        B[torch.arange(B.shape[0]), cluster_idx.long()] = 1
        v = centers.flatten()
        return B, v

    def __call__(self, tensor):
        B, v = self.clustering(tensor)
        tensor_q = torch.matmul(B, v).view(tensor.shape)

        return tensor_q


class CentroidsQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, c, T):
        b = (c[1:] + c[:-1]) / 2
        s = c[1:] - c[:-1]
        # TODO: add not symmetric (Relu) case
        g = torch.sigmoid(T * (tensor.view(-1, 1) - b))

        ctx.save_for_backward(b, s, g, T)

        y = torch.sum(s * g, dim=1) + c[0]
        return y.view(tensor.shape)

    @staticmethod
    def backward(ctx, grad_output):
        b, s, g, T = ctx.saved_tensors

        t = s * g * (1 - g)

        grad_x = T * torch.sum(t, dim=1).view(grad_output.shape)
        grad_x.mul_(grad_output)
        grad_x.clamp_(-100, 100)

        # TODO: check dimensions to handle cases where dim=1 less than 4
        grad_ck = g[:, -1].clone()
        grad_ck.sub_((T / 2) * t[:, -1])
        grad_ck.mul_(grad_output.flatten())
        grad_ck = torch.sum(grad_ck, dim=0).view(-1)

        grad_cj = g[:, :-1] - g[:, 1:]
        grad_cj.sub_((T / 2) * (t[:, 1:] + t[:, :-1]))
        grad_cj.mul_(grad_output.view(-1, 1))
        grad_cj = torch.sum(grad_cj, dim=0).view(-1)

        grad_c0 = -g[:, 0].clone()
        grad_c0.sub_((T / 2) * t[:, 0])
        grad_c0.add_(1.)
        grad_c0.mul_(grad_output.flatten())
        grad_c0 = torch.sum(grad_c0, dim=0).view(-1)

        grad_c = torch.clamp(torch.cat((grad_c0, grad_cj, grad_ck)), -1, 1)

        return grad_x, grad_c, None


class LearnedCentroidsQuantization(QuantizationBase):
    c_param_name = 'c'

    def __init__(self, module, num_bits, symmetric, tensor, temperature):
        super(LearnedCentroidsQuantization, self).__init__(module, num_bits, symmetric)
        self.soft_quant = True
        self.hard_quant = False

        with torch.no_grad():
            # Initialize k-means centroids uniformaly
            _, centers = KmeansQuantization(num_bits).clustering(tensor)
            edges, _ = centers.flatten().sort()

        self.register_buffer('centroids', edges)
        self.register_buffer('temperature', tensor.new_tensor([temperature]))
        self.register_buffer('rho', tensor.new_tensor([1.]))

        self.register_parameter(self.c_param_name, edges.new_ones(edges.shape))

        self.__create_optim_params__()

    def __call__(self, tensor):
        c = self.centroids * self.c

        # Soft quantization
        if self.soft_quant:
            T = self.temperature
            # b = (c[1:] + c[:-1]) / 2
            # s = c[1:] - c[:-1]
            #
            # y = torch.sum(s * torch.sigmoid(T * (tensor.view(-1, 1) - b)), dim=1).view(tensor.shape)
            # if self.symmetric:
            #     # In symmetric case shift central bin to 0
            #     tensor_soft = y + c[0]
            # else:
            #     # Asymmetric case after relu, avoid changing zeros in the tensor
            #     tensor_soft = torch.where(tensor > 0, y, tensor)

            tensor_soft = CentroidsQuantization.apply(tensor, c, T)

        # Hard quantization
        if self.hard_quant:
            tensor_hard = StepQuantizationSte().apply(tensor, c)

        if self.soft_quant and self.hard_quant:
            print('Hello')
            assert False
            # tensor_q = self.rho * tensor_soft + (1 - self.rho) * tensor_hard
        elif self.soft_quant:
            tensor_q = tensor_soft
        elif self.hard_quant:
            tensor_q = tensor_hard
        else:
            raise RuntimeError('quantization not defined!!!')

        return tensor_q

    def loggable_parameters(self):
        lp = super(LearnedCentroidsQuantization, self).loggable_parameters()
        return lp + [('ctr', self.centroids * self.c),
                     ('rho', self.rho),
                     ('temperature', self.temperature)]

    def __create_optim_params__(self):
        # 3 bit width configuration. Need to see how to find those parameters more easily per bit width/model/dataset
        # self.__add_optim_params__('SGD', 'imagenet', [
        #     (self.c_param_name, {'params': [self.c], 'lr': 1e-4, 'momentum': 0, 'weight_decay': 0}),
        #     (self.T_param_name, {'params': [self.T], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0})
        # ])
        self.__add_optim_params__('SGD', 'imagenet', [
            (self.c_param_name, {'params': [self.c], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0}),
            # (self.T_param_name, {'params': [self.T], 'lr': 1e-1, 'momentum': 0, 'weight_decay': 0})
        ])
        self.__add_optim_params__('Adam', 'imagenet', [
            (self.c_param_name, {'params': [self.c], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0}),
            # (self.T_param_name, {'params': [self.T], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0})
        ])
        self.__add_optim_params__('SGD', 'cifar10', [
            (self.c_param_name, {'params': [self.c], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0}),
            # (self.T_param_name, {'params': [self.T], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0})
        ])
        self.__add_optim_params__('Adam', 'cifar10', [
            (self.c_param_name, {'params': [self.c], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0}),
            # (self.T_param_name, {'params': [self.T], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0})
        ])

    @staticmethod
    def learned_parameters():
        # Use this function to control what parameters to learn
        return [
                LearnedCentroidsQuantization.c_param_name
                ]

    def clustering(self, tensor):
        tq = self.quantize(tensor)
        B_tag = torch.abs(tq.view(-1, 1) - self.c.view(1, -1))
        B = B_tag.new_zeros(B_tag.shape)
        B[torch.arange(B.shape[0]), B_tag.argmin(dim=1)] = 1

        return B, self.c


class LearnedSigmoidQuantization(QuantizationBase):
    alpha_param_name = 'alpha'
    beta_param_name = 'beta'
    b_param_name = 'b'

    def __init__(self, module, num_bits, symmetric, tensor, temperature):
        super(LearnedCentroidsQuantization, self).__init__(module, num_bits, symmetric)
        self.temperature = temperature
        # TODO: change to symmetric
        self.Y = tensor.new_tensor([0, 1, 2, 4])

        with torch.no_grad():
            b = (self.Y[1:] + self.Y[:-1]) / 2

        self.register_parameter(self.beta_param_name, tensor.new_tensor([self.num_bins / tensor.abs().max()]))
        self.register_parameter(self.alpha_param_name, 1 / self.beta)
        self.register_parameter(self.b_param_name, b)

        self.__create_optim_params__()

    def __call__(self, tensor):
        T = self.temperature
        s = self.Y[1:] - self.Y[:-1]

        temp = torch.clamp(T * (self.beta * tensor.view(-1, 1) - self.b), -10, 10)
        # Assume relu, handle zeros issue
        tensor_q = torch.where(tensor > 0,
                               self.alpha * torch.sum(s * torch.sigmoid(temp), dim=1).view(tensor.shape),
                               tensor)
        # tensor_q = LearnableSigmoidQuantization.SigmoidQuantSte().apply(tensor_q, self.alpha, self.beta, s, self.b, 1e10)

        # Hard quantization
        tensor_q = StepQuantizationSte().apply(tensor_q, self.c)

        return tensor_q

    def __create_optim_params__(self):
        self.__add_optim_params__('SGD', 'imagenet', [
            (self.alpha_param_name, {'params': [self.alpha], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0}),
            (self.beta_param_name, {'params': [self.beta], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0}),
            (self.b_param_name, {'params': [self.b], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0})
        ])

    @staticmethod
    def learned_parameters():
        return [
                LearnedSigmoidQuantization.alpha_param_name,
                LearnedSigmoidQuantization.beta_param_name,
                LearnedSigmoidQuantization.b_param_name
                ]


class LearnableDifferentiableQuantization(object):
    binning_param_name = 'tensor_binning'
    T_param_name = 'lsiq_T'

    def __init__(self, module, tensor, num_bits):
        self.num_bits = num_bits
        self.temperature = 1
        self.num_bins = int(2 ** num_bits)
        self.c = None

        with torch.no_grad():
            # Initialize B with k-means quantization
            B, _ = KmeansQuantization(num_bits).clustering(tensor)

        module.register_parameter(self.binning_param_name, nn.Parameter(B))
        self.tensor_binning = getattr(module, self.binning_param_name)

        module.register_parameter(self.T_param_name, nn.Parameter(tensor.new_tensor([1 / self.temperature])))
        self.T = getattr(module, self.T_param_name)

        self.sm = nn.Softmax(dim=1)

    def clustering(self, tensor):
        B = ArgmaxMaskSTE().apply(self.tensor_binning)
        v = torch.matmul(tensor.view(-1), B) / B.sum(dim=0)
        return B, v

    def quantize(self, tensor):
        T = 1 / self.T.abs()

        # Soft quantization
        B = self.sm(T * self.tensor_binning)
        v = torch.matmul(tensor.view(-1), B) / B.sum(dim=0)
        tensor_q = torch.matmul(B, v).view(tensor.shape)

        # Hard quantization
        tensor_q = StepQuantizationSte().apply(tensor_q, v)

        self.c = v.detach().cpu()

        return tensor_q

    def loggable_parameters(self):
        return [('c', self.c), (LearnableDifferentiableQuantization.T_param_name, 1 / self.T.detach())]

    def named_parameters(self):
        np = [
              (LearnableDifferentiableQuantization.binning_param_name, self.tensor_binning),
              (LearnableDifferentiableQuantization.T_param_name, self.T),
              ]

        named_params = [(n, p) for n, p in np if n in self.learnable_parameters_names()]
        return named_params

    def optim_parameters(self):
        pdict = [
            (LearnableDifferentiableQuantization.binning_param_name, {'params': [self.tensor_binning], 'lr': 1e-1, 'momentum': 0, 'weight_decay': 0}),
            (LearnableDifferentiableQuantization.T_param_name, {'params': [self.T], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0}),
        ]

        params = [d for n, d in pdict if n in self.learnable_parameters_names()]
        return params

    @staticmethod
    def learnable_parameters_names():
        return [
                LearnableDifferentiableQuantization.binning_param_name,
                LearnableDifferentiableQuantization.T_param_name
                ]
# non-uniform.py ends


# clipped.py starts
class ClippedUniformQuantization(UniformQuantization):
    alpha_param_name = 'alpha'

    def __init__(self, module, num_bits, symmetric, uint=False, stochastic=False, tails=False):
        super(ClippedUniformQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic,tails)

    def __call__(self, tensor):
        t_q = self.__quantize__(tensor, self.alpha)
        return t_q

    def __for_repr__(self):
        rpr = super(ClippedUniformQuantization, self).__for_repr__()
        return [(self.alpha_param_name, '{:.4f}'.format(getattr(self, self.alpha_param_name).item()))] + rpr


class FixedClipValueQuantization(ClippedUniformQuantization):
    def __init__(self, module, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(FixedClipValueQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)
        self.clip_value = kwargs['clip_value']
        self.device = kwargs['device']
        with torch.no_grad():
            self.register_buffer(self.alpha_param_name, torch.tensor([self.clip_value], dtype=torch.float32).to(self.device))


class MaxAbsStaticQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(MaxAbsStaticQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            self.register_buffer(self.alpha_param_name, tensor.new_tensor([tensor.abs().max()]))


class LearnedStepSizeQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, **kwargs):
        super(LearnedStepSizeQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            maxabs = tensor.abs().max()

        self.register_parameter(self.alpha_param_name, tensor.new_tensor([maxabs]))

        self.__create_optim_params__()

    def __create_optim_params__(self):
        # TODO: create default configuration
        self.__add_optim_params__('SGD', 'imagenet', [
            (self.alpha_param_name, {'params': [getattr(self, self.alpha_param_name)], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0})
        ])
        self.__add_optim_params__('SGD', 'cifar10', [
            (self.alpha_param_name, {'params': [getattr(self, self.alpha_param_name)], 'lr': 1e-1, 'momentum': 0, 'weight_decay': 0})
        ])

    @staticmethod
    def learned_parameters():
        return [
                LearnedStepSizeQuantization.alpha_param_name
                ]


class AngDistanceQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(AngDistanceQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        xq = self.__quantize__(x, alpha)

        norm_x = torch.norm(x)
        norm_xq = torch.norm(xq)
        cos = torch.dot(x.flatten(), xq.flatten()) / (norm_x * norm_xq)
        err = torch.acos(cos)
        return err.item()


class LpNormQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(LpNormQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        self.p = kwargs['lp']
        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        xq = self.__quantize__(x, alpha)
        err = torch.mean(torch.abs(xq - x) ** self.p)
        return err.item()


class L1NormQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(L1NormQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        N = x.numel() if self.symmetric else x[x != 0].numel()
        xq = self.__quantize__(x, alpha)
        err = torch.sum(torch.abs(xq - x)) / N
        return err.item()


class L2NormQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(L2NormQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        N = x.numel() if self.symmetric else x[x != 0].numel()
        xq = self.__quantize__(x, alpha)
        err = torch.sum(torch.abs(xq - x) ** 2) / N
        return err.item()


class L3NormQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(L3NormQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        N = x.numel() if self.symmetric else x[x != 0].numel()
        xq = self.__quantize__(x, alpha)
        err = torch.sum(torch.abs(xq - x) ** 3) / N
        return err.item()


class MseNoPriorQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(MseNoPriorQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        delta = (2 if self.symmetric else 1) * alpha / (self.num_bins - 1)
        if self.tails:
            Cx = torch.clamp(x,(-alpha if self.symmetric else 0.) - delta / 2, alpha + delta / 2)
        else:
            Cx = torch.clamp(x, -alpha if self.symmetric else 0., alpha)
        Ci = Cx - x

        N = x.numel() if self.symmetric else x[x != 0].numel()
        xq = self.__quantize__(x, alpha)

        qerr_exp = torch.sum((xq - Cx)) / N
        qerrsq_exp = torch.sum((xq - Cx) ** 2) / N
        cerr = torch.sum(Ci ** 2) / N
        mixed_err = 2 * torch.sum(Ci) * alpha * qerr_exp / N
        mse = qerrsq_exp + cerr + mixed_err
        return mse.item()


class LogLikeQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(LogLikeQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            if symmetric:
                self.b = tensor.abs().mean()
            else:
                # We need to measure b before ReLu.
                # Instead assume zero mean and multiply b after relu by 2 to approximation b before relu.
                self.b = tensor[tensor != 0].abs().mean()

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor)).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        delta = (2 if self.symmetric else 1) * alpha / (self.num_bins - 1)
        Nq = x[(x > 0) & (x <= alpha)].numel()
        Nc = x[x > alpha].numel()
        clip_err = ((x[x > alpha]- alpha) / self.b).sum() + Nc * torch.log(torch.clamp(self.b, 1e-30, 1e+30))
        q_err = Nq * np.log(np.max([delta, 1e-100]))
        # print("alpha={}, delta={}, q={}, c={}, tot={}".format(alpha,delta,q_err, clip_err.item(),clip_err.item() + q_err + add))
        return clip_err.item() + q_err


class MseUniformPriorQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(MseUniformPriorQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor), bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        N = x.numel() if self.symmetric else x[x != 0].numel()
        clip_err = torch.sum((torch.clamp(x, -alpha, alpha) - x) ** 2) / N
        quant_err = alpha ** 2 / ((3 if self.symmetric else 12) * (2 ** (2 * self.num_bits)))
        err = clip_err + quant_err
        return err.item()


class AciqGausQuantization(ClippedUniformQuantization):
    gaus_mult = {1: 1.24, 2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
    gaus_mult_positive = {1: 1.71, 2: 2.15, 3: 2.55, 4: 2.93, 5: 3.28, 6: 3.61, 7: 3.92, 8: 4.2}

    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(AciqGausQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            if self.symmetric:
                sigma = tensor.std()
                alpha_opt = self.gaus_mult[self.num_bits] * sigma
            else:
                # We need to measure std before ReLu.
                # Instead assume zero mean and multiply std after relu by 2 to approximation std before relu.
                sigma = torch.sqrt(torch.mean(tensor[tensor != 0]**2))
                alpha_opt = self.gaus_mult_positive[self.num_bits] * sigma

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([alpha_opt]))


class AciqLaplaceQuantization(ClippedUniformQuantization):
    laplace_mult = {0: 1.05, 1: 1.86, 2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
    laplace_mult_positive = {0: 1.86, 1: 2.83, 2: 3.89, 3: 5.02, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16}

    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(AciqLaplaceQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            if symmetric:
                b = tensor.abs().mean()
                alpha = self.laplace_mult[self.num_bits] * b
            else:
                # We need to measure b before ReLu.
                # Instead assume zero mean and multiply b after relu by 2 to approximation b before relu.
                b = tensor[tensor != 0].abs().mean()
                alpha = self.laplace_mult_positive[self.num_bits] * b

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([alpha]))
# clipped.py ends


# We have CnnModel file here
class CnnModel(object):
    def __init__(self, arch, use_custom_resnet, use_custom_inception, pretrained, dataset, gpu_ids, datapath, batch_size, shuffle, workers,
                 print_freq, cal_batch_size, cal_set_size, args):
        self.arch = arch
        self.use_custom_resnet = use_custom_resnet
        self.pretrained = pretrained
        self.dataset = dataset
        self.gpu_ids = gpu_ids
        self.datapath = datapath
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.workers = workers
        self.print_freq = print_freq
        self.cal_batch_size = cal_batch_size
        self.cal_set_size = cal_set_size  # TODO: pass it as cmd line argument

        # create model
        if 'resnet' in arch and use_custom_resnet:
            model = custom_resnet(arch=arch, pretrained=pretrained, depth=arch2depth(arch),
                                  dataset=dataset)
        elif 'inception_v3' in arch and use_custom_inception:
            model = custom_inception(pretrained=pretrained)
        else:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=pretrained)

        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

        torch.cuda.set_device(gpu_ids[0])
        model = model.to(self.device)

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, self.device)
                args.start_epoch = checkpoint['epoch']
                checkpoint['state_dict'] = {normalize_module_name(k): v for k, v in checkpoint['state_dict'].items()}
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        if len(gpu_ids) > 1:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if arch.startswith('alexnet') or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, gpu_ids)
            else:
                model = torch.nn.DataParallel(model, gpu_ids)

        self.model = model

        if args.bn_folding:
            print("Applying batch-norm folding ahead of post-training quantization")
            
            search_absorbe_bn(model)

        # define loss function (criterion) and optimizer
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        val_data = get_dataset(dataset, 'val', get_transform(dataset, augment=False, scale_size=299 if 'inception' in arch else None,
                               input_size=299 if 'inception' in arch else None),
                               datasets_path=datapath)
        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True)

        self.cal_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.cal_batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True)

    @staticmethod
    def __arch2depth__(arch):
        depth = None
        if 'resnet18' in arch:
            depth = 18
        elif 'resnet34' in arch:
            depth = 34
        elif 'resnet50' in arch:
            depth = 50
        elif 'resnet101' in arch:
            depth = 101

        return depth

    def evaluate_calibration(self):
        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            if not hasattr(self, 'cal_set'):
                self.cal_set = []
                # TODO: Workaround, refactor this later
                for i, (images, target) in enumerate(self.cal_loader):
                    if i * self.cal_batch_size >= self.cal_set_size:
                        break
                    images = images.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    self.cal_set.append((images, target))

            res = torch.tensor([0.]).to(self.device)
            for i in range(len(self.cal_set)):
                images, target = self.cal_set[i]
                # compute output
                output = self.model(images)
                loss = self.criterion(output, target)
                res += loss

            return res / len(self.cal_set)

    def validate(self):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(self.val_loader), batch_time, losses, top1, top5,
                                 prefix='Test: ')

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(self.val_loader):
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # compute output
                output = self.model(images)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    progress.print(i)

            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        return top1.avg
# CnnModel file ends

# We have module_wrapper here
quantization_mapping = {'max_static': MaxAbsStaticQuantization,
                        'aciq_laplace': AciqLaplaceQuantization,
                        'aciq_gaus': AciqGausQuantization,
                        'mse_uniform_prior': MseUniformPriorQuantization,
                        'mse_no_prior': MseNoPriorQuantization,
                        'ang_dis': AngDistanceQuantization,
                        'l3_norm': L3NormQuantization,
                        'l2_norm': L2NormQuantization,
                        'l1_norm': L1NormQuantization,
                        'lp_norm': LpNormQuantization,
                        'log_like': LogLikeQuantization
                        }


def is_positive(module):
    return isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6)


class ActivationModuleWrapperPost(nn.Module):
    def __init__(self, name, wrapped_module, **kwargs):
        super(ActivationModuleWrapperPost, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.bits_out = kwargs['bits_out']
        self.qtype = kwargs['qtype']
        self.post_relu = True
        self.enabled = True
        self.active = True

        if self.bits_out is not None:
            self.out_quantization = self.out_quantization_default = None

            def __init_out_quantization__(tensor):
                self.out_quantization_default = quantization_mapping[self.qtype](self, tensor, self.bits_out,
                                                                                 symmetric=(not is_positive(wrapped_module)),
                                                                                 uint=True, kwargs=kwargs)
                self.out_quantization = self.out_quantization_default
                print("ActivationModuleWrapperPost - {} | {} | {}".format(self.name, str(self.out_quantization), str(tensor.device)))

            self.out_quantization_init_fn = __init_out_quantization__

    def __enabled__(self):
        return self.enabled and self.active and self.bits_out is not None

    def forward(self, *input):
        # Uncomment to enable dump
        # torch.save(*input, os.path.join('dump', self.name + '_in' + '.pt'))

        if self.post_relu:
            out = self.wrapped_module(*input)

            # Quantize output
            if self.__enabled__():
                self.verify_initialized(self.out_quantization, out, self.out_quantization_init_fn)
                out = self.out_quantization(out)
        else:
            # Quantize output
            if self.__enabled__():
                self.verify_initialized(self.out_quantization, *input, self.out_quantization_init_fn)
                out = self.out_quantization(*input)
            else:
                out = self.wrapped_module(*input)

        # Uncomment to enable dump
        # torch.save(out, os.path.join('dump', self.name + '_out' + '.pt'))

        return out

    def get_quantization(self):
        return self.out_quantization

    def set_quantization(self, qtype, kwargs, verbose=False):
        self.out_quantization = qtype(self, self.bits_out, symmetric=(not is_positive(self.wrapped_module)),
                                      uint=True, kwargs=kwargs)
        if verbose:
            print("ActivationModuleWrapperPost - {} | {} | {}".format(self.name, str(self.out_quantization),
                                                                      str(kwargs['device'])))

    def set_quant_method(self, method=None):
        if self.bits_out is not None:
            if method == 'kmeans':
                self.out_quantization = KmeansQuantization(self.bits_out)
            else:
                self.out_quantization = self.out_quantization_default

    @staticmethod
    def verify_initialized(quantization_handle, tensor, init_fn):
        if quantization_handle is None:
            init_fn(tensor)

    def log_state(self, step, ml_logger):
        if self.__enabled__():
            if self.out_quantization is not None:
                for n, p in self.out_quantization.named_parameters():
                    if p.numel() == 1:
                        ml_logger.log_metric(self.name + '.' + n, p.item(),  step='auto')
                    else:
                        for i, e in enumerate(p):
                            ml_logger.log_metric(self.name + '.' + n + '.' + str(i), e.item(),  step='auto')


class ParameterModuleWrapperPost(nn.Module):
    def __init__(self, name, wrapped_module, **kwargs):
        super(ParameterModuleWrapperPost, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.forward_functor = kwargs['forward_functor']
        self.bit_weights = kwargs['bits_weight']
        self.bits_out = kwargs['bits_out']
        self.qtype = kwargs['qtype']
        self.enabled = True
        self.active = True
        self.centroids_hist = {}
        self.log_weights_hist = False
        self.log_weights_mse = False
        self.log_clustering = False
        self.bn = kwargs['bn'] if 'bn' in kwargs else None
        self.dynamic_weight_quantization = True
        self.bcorr_w = kwargs['bcorr_w']

        setattr(self, 'weight', wrapped_module.weight)
        delattr(wrapped_module, 'weight')
        if hasattr(wrapped_module, 'bias'):
            setattr(self, 'bias', wrapped_module.bias)
            delattr(wrapped_module, 'bias')

        if self.bit_weights is not None:
            self.weight_quantization_default = quantization_mapping[self.qtype](self, self.weight, self.bit_weights,
                                                                             symmetric=True, uint=True, kwargs=kwargs)
            self.weight_quantization = self.weight_quantization_default
            if not self.dynamic_weight_quantization:
                self.weight_q = self.weight_quantization(self.weight)
                self.weight_mse = torch.mean((self.weight_q - self.weight)**2).item()
            print("ParameterModuleWrapperPost - {} | {} | {}".format(self.name, str(self.weight_quantization),
                                                                      str(self.weight.device)))

    def __enabled__(self):
        return self.enabled and self.active and self.bit_weights is not None

    def bias_corr(self, x, xq):
        bias_q = xq.view(xq.shape[0], -1).mean(-1)
        bias_orig = x.view(x.shape[0], -1).mean(-1)
        bcorr = bias_q - bias_orig

        return xq - bcorr.view(bcorr.numel(), 1, 1, 1) if len(x.shape) == 4 else xq - bcorr.view(bcorr.numel(), 1)

    def forward(self, *input):
        w = self.weight
        if self.__enabled__():
            # Quantize weights
            if self.dynamic_weight_quantization:
                w = self.weight_quantization(self.weight)

                if self.bcorr_w:
                    w = self.bias_corr(self.weight, w)
            else:
                w = self.weight_q

        out = self.forward_functor(*input, weight=w, bias=(self.bias if hasattr(self, 'bias') else None))

        return out

    def get_quantization(self):
        return self.weight_quantization

    def set_quantization(self, qtype, kwargs, verbose=False):
        self.weight_quantization = qtype(self, self.bit_weights, symmetric=True, uint=True, kwargs=kwargs)
        if verbose:
            print("ParameterModuleWrapperPost - {} | {} | {}".format(self.name, str(self.weight_quantization),
                                                                      str(kwargs['device'])))

    def set_quant_method(self, method=None):
        if self.bit_weights is not None:
            if method is None:
                self.weight_quantization = self.weight_quantization_default
            elif method == 'kmeans':
                self.weight_quantization = KmeansQuantization(self.bit_weights)
            else:
                self.weight_quantization = self.weight_quantization_default

    # TODO: make it more generic
    def set_quant_mode(self, mode=None):
        if self.bit_weights is not None:
            if mode is not None:
                self.soft = self.weight_quantization.soft_quant
                self.hard = self.weight_quantization.hard_quant
            if mode is None:
                self.weight_quantization.soft_quant = self.soft
                self.weight_quantization.hard_quant = self.hard
            elif mode == 'soft':
                self.weight_quantization.soft_quant = True
                self.weight_quantization.hard_quant = False
            elif mode == 'hard':
                self.weight_quantization.soft_quant = False
                self.weight_quantization.hard_quant = True

    def log_state(self, step, ml_logger):
        if self.__enabled__():
            if self.weight_quantization is not None:
                for n, p in self.weight_quantization.loggable_parameters():
                    if p.numel() == 1:
                        ml_logger.log_metric(self.name + '.' + n, p.item(),  step='auto')
                    else:
                        for i, e in enumerate(p):
                            ml_logger.log_metric(self.name + '.' + n + '.' + str(i), e.item(),  step='auto')

            if self.log_weights_hist:
                ml_logger.tf_logger.add_histogram(self.name + '.weight', self.weight.cpu().flatten(),  step='auto')

            if self.log_weights_mse:
                ml_logger.log_metric(self.name + '.mse_q', self.weight_mse,  step='auto')
# module_wrapper ends here

# quantizer.py starts here

class Conv2dFunctor:
    def __init__(self, conv2d):
        self.conv2d = conv2d

    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.conv2d(*input, weight, bias, self.conv2d.stride, self.conv2d.padding,
                                         self.conv2d.dilation, self.conv2d.groups)
        return res


class LinearFunctor:
    def __init__(self, linear):
        self.linear = linear

    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.linear(*input, weight, bias)
        return res


class EmbeddingFunctor:
    def __init__(self, embedding):
        self.embedding = embedding

    def __call__(self, *input, weight, bias=None):
        res = torch.nn.functional.embedding(
            *input, weight, self.embedding.padding_idx, self.embedding.max_norm,
            self.embedding.norm_type, self.embedding.scale_grad_by_freq, self.embedding.sparse)
        return res


class OptimizerBridge(object):
    def __init__(self, optimizer, settings={'algo': 'SGD', 'dataset': 'imagenet'}):
        self.optimizer = optimizer
        self.settings = settings

    def add_quantization_params(self, all_quant_params):
        key = self.settings['algo'] + '_' + self.settings['dataset']
        if key in all_quant_params:
            quant_params = all_quant_params[key]
            for group in quant_params:
                self.optimizer.add_param_group(group)


class ModelQuantizer:
    def __init__(self, model, args, quantizable_layers, replacement_factory, optimizer_bridge=None):
        self.model = model
        self.args = args
        self.bit_weights = args.bit_weights
        self.bit_act = args.bit_act
        self.post_relu = True
        self.functor_map = {nn.Conv2d: Conv2dFunctor, nn.Linear: LinearFunctor, nn.Embedding: EmbeddingFunctor}
        self.replacement_factory = replacement_factory

        self.optimizer_bridge = optimizer_bridge

        self.quantization_wrappers = []
        self.quantizable_modules = []
        self.quantizable_layers = quantizable_layers
        self._pre_process_container(model)
        self._create_quantization_wrappers()

        # TODO: hack, make it generic
        self.quantization_params = LearnedStepSizeQuantization.learned_parameters()

    def load_state_dict(self, state_dict):
        for name, qwrapper in self.quantization_wrappers:
            qwrapper.load_state_dict(state_dict)

    def freeze(self):
        for n, p in self.model.named_parameters():
            # TODO: hack, make it more robust
            if not np.any([qp in n for qp in self.quantization_params]):
                p.requires_grad = False

    @staticmethod
    def has_children(module):
        try:
            next(module.children())
            return True
        except StopIteration:
            return False

    def _create_quantization_wrappers(self):
        for qm in self.quantizable_modules:
            # replace module by it's wrapper
            fn = self.functor_map[type(qm.module)](qm.module) if type(qm.module) in self.functor_map else None
            args = {"bits_out": self.bit_act, "bits_weight": self.bit_weights, "forward_functor": fn,
                    "post_relu": self.post_relu, "optim_bridge": self.optimizer_bridge}
            args.update(vars(self.args))
            if hasattr(qm, 'bn'):
                args['bn'] = qm.bn
            module_wrapper = self.replacement_factory[type(qm.module)](qm.full_name, qm.module,
                                                                    **args)
            setattr(qm.container, qm.name, module_wrapper)
            self.quantization_wrappers.append((qm.full_name, module_wrapper))

    def _pre_process_container(self, container, prefix=''):
        prev, prev_name = None, None
        for name, module in container.named_children():
            full_name = prefix + name
            if full_name in self.quantizable_layers:
                self.quantizable_modules.append(
                    type('', (object,), {'name': name, 'full_name': full_name, 'module': module, 'container': container})()
                )

            if self.has_children(module):
                # For container we call recursively
                self._pre_process_container(module, full_name + '.')

            prev = module
            prev_name = full_name

    def log_quantizer_state(self, ml_logger, step):
        if self.bit_weights is not None or self.bit_act is not None:
            with torch.no_grad():
                for name, qwrapper in self.quantization_wrappers:
                    qwrapper.log_state(step, ml_logger)

    def get_qwrappers(self):
        return [qwrapper for (name, qwrapper) in self.quantization_wrappers if qwrapper.__enabled__()]

    def set_clipping(self, clipping, device):  # TODO: handle device internally somehow
        qwrappers = self.get_qwrappers()
        for i, qwrapper in enumerate(qwrappers):
            qwrapper.set_quantization(FixedClipValueQuantization,
                                      {'clip_value': clipping[i], 'device': device})

    def get_clipping(self):
        clipping = []
        qwrappers = self.get_qwrappers()
        for i, qwrapper in enumerate(qwrappers):
            q = qwrapper.get_quantization()
            clip_value = getattr(q, 'alpha')
            clipping.append(clip_value.item())

        return qwrappers[0].get_quantization().alpha.new_tensor(clipping)

    class QuantMethod:
        def __init__(self, quantization_wrappers, method):
            self.quantization_wrappers = quantization_wrappers
            self.method = method

        def __enter__(self):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_method(self.method)

        def __exit__(self, exc_type, exc_val, exc_tb):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_method()

    class QuantMode:
        def __init__(self, quantization_wrappers, mode):
            self.quantization_wrappers = quantization_wrappers
            self.mode = mode

        def __enter__(self):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_mode(self.mode)

        def __exit__(self, exc_type, exc_val, exc_tb):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_mode()

    class DisableQuantizer:
        def __init__(self, quantization_wrappers):
            self.quantization_wrappers = quantization_wrappers

        def __enter__(self):
            for n, qw in self.quantization_wrappers:
                qw.active = False

        def __exit__(self, exc_type, exc_val, exc_tb):
            for n, qw in self.quantization_wrappers:
                qw.active = True

    def quantization_method(self, method):
        return ModelQuantizer.QuantMethod(self.quantization_wrappers, method)

    def quantization_mode(self, mode):
        return ModelQuantizer.QuantMode(self.quantization_wrappers, mode)

    def disable(self):
        return ModelQuantizer.DisableQuantizer(self.quantization_wrappers)
# quantizer.py ends here


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#Arguments dictionary
kwargs = {
    'arch' : 'resnet18',
    'dataset' : 'imagenet',
    'datapath' : "/workspace/code/Akash/ImageNet",
    'workers' : 4,
    'batch_size' : 256,
    'cal-batch-size' : None,
    'cal-set-size' : None,
    'print-freq' : 10,
    'resume' : '',
    'evaluate' : True,
    'pretrained' : True,
    'custom_resnet': True,
    'seed' : 0,
    'gpu_ids' : [6],
    'shuffle' : True,
    'experiment' : 'default',
    'bit_weights' : None,
    'bit_act' : None,
    'pre_relu' : True,
    'qtype' :'max_static',
    'lp' : 3.0,
    'min_method' : 'Powell',
    'maxiter' : None,
    'maxfev' : None,
    'init_method' : 'static',
    'siv' : 1. ,
    'dont_fix_np_seed' : True,
    'bcorr_w' : True,
    'tag' : 'n/a',
    'bn_folding' : True
    }

home = str(Path.home())
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name')
parser.add_argument('--datapath', metavar='DATAPATH', type=str, default=None,
                    help='dataset folder')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-cb', '--cal-batch-size', default=None, type=int, help='Batch size for calibration')
parser.add_argument('-cs', '--cal-set-size', default=None, type=int, help='Batch size for calibration')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--custom_resnet', action='store_true', help='use custom resnet implementation')
parser.add_argument('--custom_inception', action='store_true', help='use custom inception implementation')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu_ids', default=[0], type=int, nargs='+',
                    help='GPU ids to use (e.g 0 1 2 3)')
parser.add_argument('--shuffle', '-sh', action='store_true', help='shuffle data')

parser.add_argument('--experiment', '-exp', help='Name of the experiment', default='default')
parser.add_argument('--bit_weights', '-bw', type=int, help='Number of bits for weights', default=None)
parser.add_argument('--bit_act', '-ba', type=int, help='Number of bits for activations', default=None)
parser.add_argument('--pre_relu', dest='pre_relu', action='store_true', help='use pre-ReLU quantization')
parser.add_argument('--qtype', default='max_static', help='Type of quantization method')
parser.add_argument('-lp', type=float, help='p parameter of Lp norm', default=3.)

parser.add_argument('--min_method', '-mm', help='Minimization method to use [Nelder-Mead, Powell, COBYLA]', default='Powell')
parser.add_argument('--maxiter', '-maxi', type=int, help='Maximum number of iterations to minimize algo', default=None)
parser.add_argument('--maxfev', '-maxf', type=int, help='Maximum number of function evaluations of minimize algo', default=None)

parser.add_argument('--init_method', default='static',
                    help='Scale initialization method [static, dynamic, random], default=static')
parser.add_argument('-siv', type=float, help='Value for static initialization', default=1.)

parser.add_argument('--dont_fix_np_seed', '-dfns', action='store_true', help='Do not fix np seed even if seed specified')
parser.add_argument('--bcorr_w', '-bcw', action='store_true', help='Bias correction for weights', default=False)
parser.add_argument('--tag', help='Tag for logging purposes', default='n/a')
parser.add_argument('--bn_folding', '-bnf', action='store_true', help='Apply Batch Norm folding', default=False)


# TODO: refactor this
_eval_count = count(0)
_min_loss = 1e6


def evaluate_calibration_clipped(scales, model, mq):
    global _eval_count, _min_loss
    eval_count = next(_eval_count)

    mq.set_clipping(scales, model.device)
    loss = model.evaluate_calibration().item()

    if loss < _min_loss:
        _min_loss = loss

    print_freq = 20
    if eval_count % 20 == 0:
        print("func eval iteration: {}, minimum loss of last {} iterations: {:.4f}".format(
            eval_count, print_freq, _min_loss))

    return loss


def coord_descent(fun, init, args, **kwargs):
    maxiter = kwargs['maxiter']
    x = init.copy()

    def coord_opt(alpha, scales, i):
        if alpha < 0:
            result = 1e6
        else:
            scales[i] = alpha
            result = fun(scales)

        return result

    nfev = 0
    for j in range(maxiter):
        for i in range(len(x)):
            print("Optimizing variable {}".format(i))
            r = opt.minimize_scalar(lambda alpha: coord_opt(alpha, x, i))
            nfev += r.nfev
            opt_alpha = r.x
            x[i] = opt_alpha

        if 'callback' in kwargs:
            kwargs['callback'](x)

    res = opt.OptimizeResult()
    res.x = x
    res.nit = maxiter
    res.nfev = nfev
    res.fun = np.array([r.fun])
    res.success = True

    return res


def main(args):
    # Fix the seed
    random.seed(args.seed)
    if not args.dont_fix_np_seed:
        np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    args.qtype = 'max_static'
    # create model
    # Always enable shuffling to avoid issues where we get bad results due to weak statistics
    custom_resnet = True
    custom_inception = True
    inf_model = CnnModel(args.arch, custom_resnet, custom_inception, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                         batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                         cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size, args=args)

    layers = []
    # TODO: make it more generic
    if 'inception' in args.arch and args.custom_inception:
        first = 3
        last = -1
    else:
        first = 1
        last = -1
    if args.bit_weights is not None:
        layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.Conv2d)][first:last]
    if args.bit_act is not None:
        layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.ReLU)][first:last]
    if args.bit_act is not None and 'mobilenet' in args.arch:
        layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.ReLU6)][first:last]

    replacement_factory = {nn.ReLU: ActivationModuleWrapperPost,
                           nn.ReLU6: ActivationModuleWrapperPost,
                           nn.Conv2d: ParameterModuleWrapperPost}

    mq = ModelQuantizer(inf_model.model, args, layers, replacement_factory)
    maxabs_loss = inf_model.evaluate_calibration()
    print("max loss: {:.4f}".format(maxabs_loss.item()))
    max_point = mq.get_clipping()

    # evaluate
    maxabs_acc = 0#inf_model.validate()
    data = {'max': {'alpha': max_point.cpu().numpy(), 'loss': maxabs_loss.item(), 'acc': maxabs_acc}}

    del inf_model
    del mq

    def eval_pnorm(p):
        args.qtype = 'lp_norm'
        args.lp = p
        # Fix the seed
        random.seed(args.seed)
        if not args.dont_fix_np_seed:
            np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        inf_model = CnnModel(args.arch, custom_resnet, custom_inception, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                             batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                             cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size, args=args)

        mq = ModelQuantizer(inf_model.model, args, layers, replacement_factory)
        loss = inf_model.evaluate_calibration()
        point = mq.get_clipping()

        # evaluate
        acc = inf_model.validate()

        del inf_model
        del mq

        return point, loss, acc

    def eval_pnorm_on_calibration(p):
        args.qtype = 'lp_norm'
        args.lp = p
        # Fix the seed
        random.seed(args.seed)
        if not args.dont_fix_np_seed:
            np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        inf_model = CnnModel(args.arch, custom_resnet, custom_inception, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                             batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                             cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size, args=args)

        mq = ModelQuantizer(inf_model.model, args, layers, replacement_factory)
        loss = inf_model.evaluate_calibration()
        point = mq.get_clipping()

        del inf_model
        del mq

        return point, loss

    ps = np.linspace(2, 4, 10)
    losses = []
    for p in tqdm(ps):
        point, loss = eval_pnorm_on_calibration(p)
        losses.append(loss.item())
        print("(p, loss) - ({}, {})".format(p, loss.item()))

    # Interpolate optimal p
    z = np.polyfit(ps, losses, 2)
    y = np.poly1d(z)
    p_intr = y.deriv().roots[0]
    # loss_opt = y(p_intr)
    print("p intr: {:.2f}".format(p_intr))

    lp_point, lp_loss, lp_acc = eval_pnorm(p_intr)

    print("loss p intr: {:.4f}".format(lp_loss.item()))
    print("acc p intr: {:.4f}".format(lp_acc))

    global _eval_count, _min_loss
    _min_loss = lp_loss.item()

    init = lp_point

    args.qtype = 'lp_norm'
    args.lp = p_intr
    # Fix the seed
    random.seed(args.seed)
    if not args.dont_fix_np_seed:
        np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # if enable_bcorr:
    #     args.bcorr_w = True
    inf_model = CnnModel(args.arch, custom_resnet, custom_inception, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                         batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                         cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size, args=args)

    mq = ModelQuantizer(inf_model.model, args, layers, replacement_factory)

    # run optimizer
    min_options = {}
    if args.maxiter is not None:
        min_options['maxiter'] = args.maxiter
    if args.maxfev is not None:
        min_options['maxfev'] = args.maxfev

    _iter = count(0)

    def local_search_callback(x):
        it = next(_iter)
        mq.set_clipping(x, inf_model.device)
        loss = inf_model.evaluate_calibration()
        print("\n[{}]: Local search callback".format(it))
        print("loss: {:.4f}\n".format(loss.item()))
        print(x)

        # evaluate
        acc = inf_model.validate()

    args.min_method = "Powell"
    method = coord_descent if args.min_method == 'CD' else args.min_method
    res = opt.minimize(lambda scales: evaluate_calibration_clipped(scales, inf_model, mq), init.cpu().numpy(),
                       method=method, options=min_options, callback=local_search_callback)

    print(res)

    scales = res.x
    mq.set_clipping(scales, inf_model.device)
    loss = inf_model.evaluate_calibration()

    # evaluate
    acc = inf_model.validate()
    data['powell'] = {'alpha': scales, 'loss': loss.item(), 'acc': acc}

    # save scales
    f_name = "scales_{}_W{}A{}.pkl".format(args.arch, args.bit_weights, args.bit_act)
    f = open(os.path.join(proj_root_dir, 'data', f_name), 'wb')
    pickle.dump(data, f)
    f.close()
    print("Data saved to {}".format(f_name))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.cal_batch_size is None:
        args.cal_batch_size = args.batch_size
    if args.cal_batch_size > args.batch_size:
        print("Changing cal_batch_size parameter from {} to {}".format(args.cal_batch_size, args.batch_size))
        args.cal_batch_size = args.batch_size
    if args.cal_set_size is None:
        args.cal_set_size = args.batch_size

    main(args)