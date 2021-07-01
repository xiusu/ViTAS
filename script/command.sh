#!/bin/bash

job_name=$1
train_gpu=$2
num_node=$3
command=$4
total_process=$((train_gpu*num_node))

mkdir -p log

now=$(date +"%Y%m%d_%H%M%S")

# nohup 
GLOG_vmodule=MemcachedClient=-1 \
srun --partition=VA \
--mpi=pmi2 -n$total_process \
--gres=gpu:$train_gpu \
--ntasks-per-node=$train_gpu \
--cpus-per-task=4 \
--job-name=$job_name \
--kill-on-bad-exit=1 \
$command  2>&1|tee -a log/$job_name.log &


