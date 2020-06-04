#!/usr/local/bin/bash
for backend in gtcuda gtmc gtx86 numpy debug
do
  for nxy in 32 64 128 256 512 1024
  do
    for stencil_name in vertical_advection horizontal_diffusion
    do
      python ./perftest.py $nxy 10 $stencil_name $backend out.txt
    done
  done
done
