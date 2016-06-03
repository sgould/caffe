#!/usr/bin/env bash

TRAIN=1
TOOLS=./build/tools

if [ $TRAIN != 0 ] 
then
    $TOOLS/caffe train \
        --solver=sandbox/cifar10/cifar10_quick_smux_solver.prototxt || exit 1
fi

# reduce learning rate by factor of 10 after 8 epochs
if [ $TRAIN != 0 ]
then
    $TOOLS/caffe train \
        --solver=sandbox/cifar10/cifar10_quick_smux_solver_lr1.prototxt \
        --snapshot=sandbox/cifar10/cifar10_quick_smux_iter_4000.solverstate.h5 || exit 1
fi

# just test
if [ $TRAIN == 0 ]
then
    $TOOLS/caffe test \
        --model=sandbox/cifar10/cifar10_quick_smux_train_test.prototxt \
        --weights=sandbox/cifar10/cifar10_quick_smux_iter_5000.caffemodel.h5 || exit 1
fi
