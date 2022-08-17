#!/bin/bash

#SBATCH --job-name=torch-train
#SBATCH --partition=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --time=0:45:00
#SBATCH --mem=16G

# Load modules required for runtime
module load CUDA
module load languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch
module load languages/intel/2017.01
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
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
options="examples/detect.py"

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