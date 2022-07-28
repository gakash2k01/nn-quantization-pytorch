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

# We have CnnModel file here
from utils.data import get_dataset
from utils.preprocess import get_transform
from utils.meters import AverageMeter, ProgressMeter, accuracy
from models.resnet import resnet as custom_resnet
from models.inception import inception_v3 as custom_inception
from utils.misc import normalize_module_name, arch2depth

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
            from utils.absorb_bn import search_absorbe_bn
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
from quantization.methods.clipped_uniform import AngDistanceQuantization, L3NormQuantization, L2NormQuantization, \
    LpNormQuantization, L1NormQuantization
from quantization.methods.clipped_uniform import MaxAbsStaticQuantization, AciqLaplaceQuantization, \
    AciqGausQuantization, LogLikeQuantization
from quantization.methods.clipped_uniform import MseNoPriorQuantization, MseUniformPriorQuantization
from quantization.methods.non_uniform import KmeansQuantization

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
import torch
import torch.nn as nn
import numpy as np
from itertools import count
from quantization.methods.clipped_uniform import LearnedStepSizeQuantization
from quantization.methods.non_uniform import LearnableDifferentiableQuantization, LearnedCentroidsQuantization
from quantization.methods.clipped_uniform import FixedClipValueQuantization
# from utils.absorb_bn import is_absorbing, is_bn


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
