#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=4 python main.py \
--model=resnet \
--loss=bce \
--T=3 \
