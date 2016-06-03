#!/usr/bin/env bash

TRAIN=1
TOOLS=./build/tools

if [ $TRAIN != 0 ] 
then
    $TOOLS/caffe train \
	--solver=sandbox/cifar10/cifar10_full_smux_solver.prototxt
fi

# reduce learning rate by factor of 10
if [ $TRAIN != 0 ] 
then
    $TOOLS/caffe train \
	--solver=sandbox/cifar10/cifar10_full_smux_solver_lr1.prototxt \
	--snapshot=sandbox/cifar10/cifar10_full_smux_iter_60000.solverstate.h5
fi

# reduce learning rate by factor of 10
if [ $TRAIN != 0 ] 
then
    $TOOLS/caffe train \
	--solver=sandbox/cifar10/cifar10_full_smux_solver_lr2.prototxt \
	--snapshot=sandbox/cifar10/cifar10_full_smux_iter_65000.solverstate.h5
fi

# just test
if [ $TRAIN == 0 ]
then
    $TOOLS/caffe test \
        --model=sandbox/cifar10/cifar10_full_smux_train_test.prototxt \
        --weights=sandbox/cifar10/cifar10_full_smux_iter_70000.caffemodel.h5 || exit 1
fi
