#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=0 python3 imagenet_orthReg.py \
-a=vgg16 \
--batch-size=128 \
--epochs=100 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=1e-4 \
--lr=0.01 \
--lr-method=step \
--lr-steps=30 \
--lr-gamma=0.1 \
--dataset-root=/raid/Lei_Data/imageNet/input_torch/ \
--dataset=folder \
--norm=No \
--orthReg=1e-5 \
--seed=1 \
$@
