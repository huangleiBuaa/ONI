#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=5 python main.py \
--model=resnet_ONI \
--loss=bce \
--T=3 \
--NScale=1 \
