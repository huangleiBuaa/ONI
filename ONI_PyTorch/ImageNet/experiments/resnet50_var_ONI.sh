#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=5 python3 imagenet.py \
-a=resnet_var_ONI50 \
--batch-size=256 \
--epochs=100 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=2e-4 \
--lr=0.1 \
--lr-method=step \
--lr-steps=30 \
--lr-gamma=0.1 \
--normConv=ONI \
--normConv-cfg=T=2,norm_groups=1,NScale=1.414,adjustScale=True \
--dataset-root=/raid/Lei_Data/imageNet/input_torch/ \
--dataset=folder \
--norm=BN \
--seed=1 \
$@
