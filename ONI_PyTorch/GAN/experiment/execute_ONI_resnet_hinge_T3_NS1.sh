#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=7 python main.py \
--model=resnet_ONI \
--loss=hinge \
--T=3 \
--NScale=1 \
