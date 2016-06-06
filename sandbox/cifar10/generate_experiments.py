#!/usr/bin/env python
# Stephen Gould <stephen.gould@anu.edu.au>
#
"""
Script to generate experiment files from templates.
"""

import os
import sys
import numpy as np
from string import Template

def conv1layer(id):
    """Create a conv1 layer"""

    return Template("""layer {
  name: "conv1${id}"
  type: "Convolution"
  bottom: "data"
  top: "conv1${id}"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 32
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.0001
    }
    bias_filler {
      type: "constant"
    }
  }
}
""").substitute(id=id)

def conv2layer(id):
    """Create a conv2 layer"""

    return Template("""layer {
  name: "conv2${id}"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2${id}"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 32
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}
""").substitute(id=id)

def conv3layer(id):
    """Create a conv3 layer."""

    return Template("""layer {
  name: "conv3${id}"
  type: "Convolution"
  bottom: "norm2"
  top: "conv3${id}"
  convolution_param {
    num_output: 64
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}
""").substitute(id=id)

def mux1layer(operation, count):
    """
    Create a mux1 layer.
    
    :param operation: 3 for rand or 2 for max
    :param count: number of layers to multiplex
    """

    bottom = ''
    for id in range(count):
        bottom += '  bottom: "conv1' + chr(ord('a') + id) + '"\n'

    return Template("""layer {
  name: "smux1"
  type: "Eltwise"
  top: "conv1"
${bottom}
  eltwise_param {
    operation: ${op}
  }
}
""").substitute(op=operation, bottom=bottom)

def muxlayer(layer, trainop, testop, count):
    """
    Create a mux layer.

    :param layer: layer number, 1 ...
    :param trainop: 3 for rand, 2 for max, 1 for sum
    :param testop: 3 for rand, 2 for max, 1 for sum
    :param count: number of layers to multiplex
    """

    bottom = ''
    for id in range(count):
        bottom += '  bottom: "conv' + str(layer) + chr(ord('a') + id) + '"\n'

    train = Template("""layer {
  name: "mux_train${layer}"
  include {
    phase: TRAIN
  }
  type: "Eltwise"
  top: "conv${layer}"
${bottom}
  eltwise_param {
    operation: ${trainop}
  }
}

layer {
  name: "relu${layer}"
  include {
    phase: TRAIN
  }
  type: "ReLU"
  bottom: "conv${layer}"
  top: "conv${layer}"
}
""").substitute(layer=layer, bottom=bottom, trainop=trainop)

    test = ''
    for id in range(count):
        test += Template("""layer {
  name: "relu${layer}${id}"
  include {
    phase: TEST
  }
  type: "ReLU"
  bottom: "conv${layer}${id}"
  top: "conv${layer}${id}"
}\n""").substitute(layer=layer, id=chr(ord('a') + id))

    coeff = ('    coeff: ' + str(1.0 / count) + '\n') * count
    test += Template("""
layer {
  name: "mux_test${layer}"
  include {
    phase: TEST
  }
  type: "Eltwise"
  top: "conv${layer}"
${bottom}
  eltwise_param {
    operation: ${testop}
${coeff}
  }
}
""").substitute(layer=layer, bottom=bottom, testop=testop, coeff=coeff)

    return train + test

# read model template
f = open('cifar10_full_train_test.template', 'r')
tmpl = f.read()
f.close()

# create basic model
conv1 = ''
conv2 = ''
conv3 = ''
for id in range(2):
    conv1 += conv1layer(chr(ord('a') + id))
    conv2 += conv2layer(chr(ord('a') + id))
    conv3 += conv3layer(chr(ord('a') + id))

out = Template(tmpl).substitute(modelprefix='test',
    conv1layers=conv1, mux1layers=mux1layer(3, 2),
    conv2layers=conv2, mux2layers=muxlayer(2, 3, 1, 2),
    conv3layers=conv3, mux3layers=muxlayer(3, 3, 1, 2))

print(out)
