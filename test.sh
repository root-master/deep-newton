#!/bin/bash
## declare an array variable
declare -a m_vec=(5 10 15 20 40)
declare -a mini_batch_vec=(64 128 256 512 1024)
for m in ${m_vec[@]}
do
	for minibatch in ${mini_batch_vec[@]}
	do
		echo $m and $minibatch 
	done
done
