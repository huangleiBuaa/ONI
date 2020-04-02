#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=2 python main.py \
--model=dcgan_ONI \
--loss=bce \
--T=2 \
--NScale=1 \
