#!/bin/bash

#SBATCH -J cnn-4L_hw
#SBATCH -n32
#SBATCH --mem=80G
#SBATCH --qos=lowest
#SBATCH -p batch

echo 'Your job is running on node(s):'
echo $SLURM_JOB_NODELIST
echo 'Cores per node:'
echo $SLURM_TASKS_PER_NODE

rm -rf /scratch/$USER/cnn-4L

module load xilinx/vivado/2021.2 tapa
rsync -r ~/projects/Llama2Lite/benchmark/cnn-4L /scratch/$USER/ --exclude ".git"
cd /scratch/$USER/cnn-4L/sequential/
make build_hw
rsync -r /scratch/$USER/cnn-4L/sequential/work.out/ ~/projects/Llama2Lite/benchmark/cnn-4L/sequential/work.out/ --exclude ".git" --remove-source-files
rsync -r /scratch/$USER/cnn-4L/sequential/bitstreams/ ~/projects/Llama2Lite/benchmark/cnn-4L/sequential/bitstreams/ --exclude ".git" --remove-source-files
