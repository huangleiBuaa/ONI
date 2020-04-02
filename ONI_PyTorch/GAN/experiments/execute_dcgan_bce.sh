#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=2 python main.py \
--model=dcgan \
--loss=bce \
--T=3 \
