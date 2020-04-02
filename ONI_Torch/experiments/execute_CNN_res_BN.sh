#!/bin/bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=1 th CNN_Cifar10.lua -model res_BN -depth 110 -dropout 0 -m_perGroup 64 -seed 1 -batchSize 128 -learningRate 0.1 -weightDecay 0.0001 -widen_factor 1 -noNesterov 0 -max_epoch 160 -N_scale 0.8 -nIter 5 -learningRateDecayRatio 0.1 -epoch_step '{80,120}'
