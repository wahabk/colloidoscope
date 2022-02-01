#!/bin/bash

#SBATCH --job-name=torch-train
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=0:05:00
#SBATCH --mem=16G

# Load modules required for runtime
module load lang/cuda/11.1
module load lang/python/anaconda/3.8.8-2021.05-torch
# export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
# export CUDA_VISIBLE_DEVICES=0
nvidia-smi
nvcc --version

# module list

#! Mail to user if job aborts
#SBATCH --mail-type=FAIL

echo 'My torch test'
source activate #!activate conda
conda activate colloids

#! application name
application="python3"

#! Run options for the application
options="scripts/bp1.py"

#####################################
### You should not have to change ###
###   anything below this line    ###
#####################################
#! change the working directory
#! (default is home directory)

cd $SLURM_SUBMIT_DIR
# cd '/mnt/storage/home/ak18001/code/deepcolloid'

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

#! Run the threaded exe
$application $options