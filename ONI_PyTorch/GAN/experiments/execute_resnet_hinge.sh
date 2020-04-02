#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=3 python main.py \
--model=resnet \
--loss=hinge \
--T=3 \
