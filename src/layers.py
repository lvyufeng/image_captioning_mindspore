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
