import math
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import constexpr
from mindspore.common.initializer import initializer, Normal, Uniform, HeUniform, _calculate_fan_in_and_fan_out

@constexpr
def compute_kernel_size(inp_shape, output_size):
    kernel_width, kernel_height = inp_shape[2], inp_shape[3]
    if isinstance(output_size, int):
        kernel_width = math.ceil(kernel_width / output_size) 
        kernel_height = math.ceil(kernel_height / output_size)
    elif isinstance(output_size, list) or isinstance(output_size, tuple):
        kernel_width = math.ceil(kernel_width / output_size[0]) 
        kernel_height = math.ceil(kernel_height / output_size[1])
    return (kernel_width, kernel_height)

class AdaptiveMaxPool2d(nn.Cell):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.MaxPool(kernel_size, kernel_size)(x)

class AdaptiveAvgPool2d(nn.Cell):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.AvgPool(kernel_size, kernel_size)(x)

class MaxPool2d(nn.Cell):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.max_pool = ops.MaxPool(kernel_size, stride)
        self.use_pad = padding != 0
        if isinstance(padding, tuple):
            assert len(padding) == 2
            paddings = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        elif isinstance(padding, int):
            paddings = ((0, 0),) * 2 + ((padding, padding),) * 2
        else:
            raise ValueError('padding should be a tuple include 2 numbers or a int number')
        self.pad = ops.Pad(paddings)
    
    def construct(self, x):
        if self.use_pad:
            x = self.pad(x)
        return self.max_pool(x)

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, weight_init=None, bias_init=None, has_bias=True, activation=None):
        if weight_init is None:
            weight_init = initializer(HeUniform(math.sqrt(5)), (out_channels, in_channels))
        if bias_init is None:
            fan_in, _ = _calculate_fan_in_and_fan_out((out_channels, in_channels))
            bound = 1 / math.sqrt(fan_in)
            bias_init = initializer(Uniform(bound), (out_channels))
        super().__init__(in_channels, out_channels, weight_init=weight_init, bias_init=bias_init, has_bias=has_bias, activation=activation)

class CrossEntropyLoss(nn.Cell):
    reduction_list = ['sum', 'mean', 'none']
    def __init__(self, weight=None, ignore_index:int=-100, reduction:str='mean', label_smoothing:float=0.0):        
        super().__init__()
        if label_smoothing > 1.0 or label_smoothing < 0.0:
            raise ValueError(f'label_smoothing value must in range [0.0, 1.0], '
                             f'but get {label_smoothing}')
        
        if reduction not in self.reduction_list:
            raise ValueError(f'Unsupported reduction {reduction}')
        
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def construct(self, input, target):
        return cross_entropy(input, target, self.weight, self.ignore_index, self.reduction, self.label_smoothing)

def log_softmax(input, axis=-1):
    return ops.log(ops.Softmax(axis)(input))

def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    if input.size == target.size:
        return _cross_entropy(input, target, weight, reduction, label_smoothing)
    return nll_loss(log_softmax(input, 1), target, weight, ignore_index, reduction, label_smoothing)

def _cross_entropy(input, target, weight=None, reduction='mean', label_smoothing=0.0):
    class_dim = 0 if input.ndim == 1 else 1
    n_classes = input.shape[class_dim]
    input = log_softmax(input, class_dim)
    if label_smoothing > 0.0:
        target = target * (1 - label_smoothing) + label_smoothing / n_classes
    
    if weight is None:
        weight = ops.ones_like(input)

    if reduction == 'mean':
        return -(input * target * weight).sum() / (input.size / n_classes)
    elif reduction == 'sum':
        return -(input * target * weight).sum()
    else:
        return -(input * target * weight).sum(class_dim)

def nll_loss(input, target, weight=None, ignore_index=None, reduction='mean', label_smoothing=0.0):
    ndim = input.ndim
    if ndim == 2:
        ret = _nll_loss(input, target, -1, weight, ignore_index, reduction, label_smoothing)
    elif input.ndim == 4:
        ret = _nll_loss(input, target, 1, weight, ignore_index, reduction, label_smoothing)
    else:
        # ndim == 3 or ndim > 4
        n = input.shape[0]
        c = input.shape[1]
        out_size = (n,) + input.shape[2:]
        input = input.view(n, c, 1, -1)
        target = target.view(n, 1, -1)
        if reduction != 'none':
            ret = _nll_loss(input, target, 1, weight, ignore_index, reduction, label_smoothing)
        else:
            ret = _nll_loss(input, target, 1, weight, ignore_index, label_smoothing=label_smoothing)
            ret = ret.view(out_size)
    return ret

def _nll_loss(input, target, target_dim=-1, weight=None, ignore_index=None, reduction='none', label_smoothing=0.0):
    if target.ndim == input.ndim - 1:
        target = target.expand_dims(target_dim)
    nll_loss = -ops.gather_d(input, target_dim, target)
    smooth_loss = -input.sum(axis=target_dim, keepdims=True)
    if weight is not None:
        loss_weights = ops.gather(weight, target, 0)
        nll_loss = nll_loss * loss_weights
    else:
        loss_weights = ops.ones_like(nll_loss)
    if ignore_index is not None:
        non_pad_mask = ops.equal(target, ignore_index)
        nll_loss = nll_loss.masked_fill(non_pad_mask, 0.)
        loss_weights = loss_weights.masked_fill(non_pad_mask, 0.)
        smooth_loss = smooth_loss.masked_fill(non_pad_mask, 0.)

    nll_loss = nll_loss.squeeze(target_dim)
    smooth_loss = smooth_loss.squeeze(target_dim)

    if reduction == 'sum':
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == 'mean':
        nll_loss = nll_loss.sum() / loss_weights.sum()
        smooth_loss = smooth_loss.mean()
    
    eps_i = label_smoothing / input.shape[target_dim]
    loss = (1. - label_smoothing) * nll_loss + eps_i * smooth_loss

    return loss
