#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=2 python3 imagenet.py \
-a=vgg_ONI16 \
--batch-size=128 \
--epochs=100 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=1e-4 \
--lr=0.05 \
--lr-method=step \
--lr-steps=30 \
--lr-gamma=0.1 \
--dataset-root=/raid/Lei_Data/imageNet/input_torch/ \
--dataset=folder \
--normConv=ONI \
--normConv-cfg=T=5,norm_groups=1,NScale=1.414,adjustScale=True,ONIRow_Fix=True \
--norm=No \
--seed=1 \
$@
