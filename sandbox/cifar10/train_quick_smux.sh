#!/usr/bin/env bash

TRAIN=1
TOOLS=./build/tools

if [ $TRAIN != 0 ] 
then
    $TOOLS/caffe train \
        --solver=examples/cifar10/cifar10_quick_smux_solver.prototxt
fi

# reduce learning rate by factor of 10 after 8 epochs
if [ $TRAIN != 0]
then
    $TOOLS/caffe train \
        --solver=examples/cifar10/cifar10_quick_smux_solver_lr1.prototxt \
        --snapshot=examples/cifar10/cifar10_quick_smux_iter_4000.solverstate.h5
fi

# just test
if [ $TRAIN == 0 ]
then
    $TOOLS/caffe test \
        --model=examples/cifar10/cifar10_quick_smux_train_test.prototxt \
        --weights=examples/cifar10/cifar10_quick_smux_iter_5000.caffemodel.h5
fi
