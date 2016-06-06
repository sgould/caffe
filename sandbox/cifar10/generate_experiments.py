#!/usr/bin/env python
# Stephen Gould <stephen.gould@anu.edu.au>
#
"""
Script to generate experiment files from templates.
"""

import os, sys, stat
import numpy as np
from string import Template

SUM = 1
MAX = 2
RND = 3

def conv1layers(count = 0):
    """Create a conv1 layers"""

    tmpl = Template("""layer {
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
""")

    if count == 0:
        return tmpl.substitute(id='')

    conv = ''
    for id in range(count):
        conv += tmpl.substitute(id=chr(ord('a') + id))
    return conv

def conv2layers(count):
    """Create a conv2 layers."""

    tmpl = Template("""layer {
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
""")

    if count == 0:
        return tmpl.substitute(id='')

    conv = ''
    for id in range(count):
        conv += tmpl.substitute(id=chr(ord('a') + id))
    return conv

def conv3layers(count):
    """Create a conv3 layers."""

    tmpl = Template("""layer {
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
""")

    if count == 0:
        return tmpl.substitute(id='')

    conv = ''
    for id in range(count):
        conv += tmpl.substitute(id=chr(ord('a') + id))
    return conv


def mux1layers(operation, count):
    """
    Create mux1 layers.

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

def muxlayers(level, trainop, testop, count):
    """
    Create mux layers for arbitrary level.

    :param level: level number, 1 ...
    :param trainop: 3 for rand, 2 for max, 1 for sum
    :param testop: 3 for rand, 2 for max, 1 for sum
    :param count: number of convolutions to multiplex
    """

    if count == 0:
        return Template("""layer {
  name: "relu${level}"
  type: "ReLU"
  bottom: "conv${level}"
  top: "conv${level}"
}
""").substitute(level=level)

    bottom = ''
    for id in range(count):
        bottom += '  bottom: "conv' + str(level) + chr(ord('a') + id) + '"\n'

    train = Template("""layer {
  name: "mux_train${level}"
  include {
    phase: TRAIN
  }
  type: "Eltwise"
  top: "conv${level}"
${bottom}
  eltwise_param {
    operation: ${trainop}
  }
}

layer {
  name: "relu${level}"
  include {
    phase: TRAIN
  }
  type: "ReLU"
  bottom: "conv${level}"
  top: "conv${level}"
}
""").substitute(level=level, bottom=bottom, trainop=trainop)

    test = ''
    for id in range(count):
        test += Template("""layer {
  name: "relu${level}${id}"
  include {
    phase: TEST
  }
  type: "ReLU"
  bottom: "conv${level}${id}"
  top: "conv${level}${id}"
}\n""").substitute(level=level, id=chr(ord('a') + id))

    coeff = ('    coeff: ' + str(1.0 / count) + '\n') * count
    test += Template("""
layer {
  name: "mux_test${level}"
  include {
    phase: TEST
  }
  type: "Eltwise"
  top: "conv${level}"
${bottom}
  eltwise_param {
    operation: ${testop}
${coeff}
  }
}
""").substitute(level=level, bottom=bottom, testop=testop, coeff=coeff)

    return train + test

# read network template
f = open('cifar10_full_train_test.template', 'r')
net_tmpl = f.read()
f.close()

# read training templates
script_names = ['cifar10_full_solver.prototxt', 'cifar10_full_solver_lr1.prototxt',
                'cifar10_full_solver_lr2.prototxt', 'train_full.sh']
script_tmpls = {}
for name in script_names:
    with open(name + '.template', 'r') as f:
        script_tmpls[name] = f.read()

# DEBUG: create basic model
out = Template(net_tmpl).substitute(modelprefix='test',
    conv1layers=conv1layers(2), mux1layers=mux1layers(3, 2),
    conv2layers=conv2layers(2), mux2layers=muxlayers(2, 3, 1, 2),
    conv3layers=conv3layers(2), mux3layers=muxlayers(3, 3, 1, 2))
#print(out)

def create_experiment(expr, net):
    """Creates an experiment directory."""

    if not os.path.exists(expr):
        print('Creating experiment directory {}...'.format(expr))
        os.makedirs(expr)
        with open(expr + '/cifar10_full_train_test.prototxt', 'w') as f:
            f.write(net)
        for name, tmpl in script_tmpls.iteritems():
            with open(expr + '/' + name, 'w') as f:
                f.write(Template(tmpl).substitute(expr=expr))
        os.chmod(expr + '/train_full.sh', stat.S_IRWXU)
    else:
        print('WARNING: {} directory already exists'.format(expr))

# generate baseline model
create_experiment('baseline', Template(net_tmpl).substitute(modelprefix='baseline',
    conv1layers=conv1layers(0), mux1layers='',
    conv2layers=conv2layers(0), mux2layers=muxlayers(2, 0, 0, 0),
    conv3layers=conv3layers(0), mux3layers=muxlayers(3, 0, 0, 0)))

for count in range(2, 5):
    # generate rnd2sum model
    expr = 'rnd{}sum'.format(count)
    create_experiment(expr, Template(net_tmpl).substitute(modelprefix=expr,
        conv1layers=conv1layers(count), mux1layers=mux1layers(RND, count),
        conv2layers=conv2layers(count), mux2layers=muxlayers(2, RND, SUM, count),
        conv3layers=conv3layers(count), mux3layers=muxlayers(3, RND, SUM, count)))

    # generate rnd2rnd model
    expr = 'rnd{}rnd'.format(count)
    create_experiment(expr, Template(net_tmpl).substitute(modelprefix=expr,
        conv1layers=conv1layers(count), mux1layers=mux1layers(RND, count),
        conv2layers=conv2layers(count), mux2layers=muxlayers(2, RND, RND, count),
        conv3layers=conv3layers(count), mux3layers=muxlayers(3, RND, RND, count)))

    # generate max2max model
    expr = 'max{}max'.format(count)
    create_experiment(expr, Template(net_tmpl).substitute(modelprefix=expr,
        conv1layers=conv1layers(count), mux1layers=mux1layers(MAX, count),
        conv2layers=conv2layers(count), mux2layers=muxlayers(2, MAX, MAX, count),
        conv3layers=conv3layers(count), mux3layers=muxlayers(3, MAX, MAX, count)))
