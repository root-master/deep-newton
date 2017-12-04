#!/bin/bash
declare -a m_vec=(5 10 15 20)
declare -a mini_batch_vec=(128 256 512 1024)
for m in ${m_vec[@]}
do
	for minibatch in ${mini_batch_vec[@]}
	do
		python test_tmp.py -m=$m -batch=$minibatch
	done
done
