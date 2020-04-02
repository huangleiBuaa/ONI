#!/bin/bash
cd "$(dirname $0)/.."
methods=(P0_WN_scale)
depths=(8 11 14)
wfs=(2 4 6) 
learningRates=(0.02)
m_perGroup=384
nIter=5
N_scale=1.414
weightDecay=0
seed=1
maxEpoch=160
eStep="{80,120}"


n=${#methods[@]}
m=${#depths[@]}
f=${#wfs[@]}
g=${#learningRates[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do
        for ((s=0;s<$g;++s))
        do

    	echo "methods=${methods[$i]}"
    	echo "depths=${depths[$j]}"
   	    echo "wf=${wfs[$k]}"
   	    echo "learningRate=${learningRates[$s]}"
   CUDA_VISIBLE_DEVICES=2	th CNN_Cifar10.lua -model ${methods[$i]} -depth ${depths[$j]} -widen_factor ${wfs[$k]} -max_epoch ${maxEpoch} -weightDecay ${weightDecay} -N_scale ${N_scale} -seed ${seed} -learningRate ${learningRates[$s]} -m_perGroup ${m_perGroup} -nIter ${nIter} -epoch_step ${eStep}
        done
      done
   done
done
