#!/bin/bash -l
#
#SBATCH -J channel_derivatives      # name of your job   
#SBATCH -p normal                   # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 5-00:00:00               # time in d-hh:mm:ss you want to reserve for the job
#SBATCH -n 2                        # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH --mem=130G          
#SBATCH -o channel_derivatives.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e channel_derivatives.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM
#SBATCH --mail-user=b.j.h.r.reijnders@uu.nl 
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

cd /nethome/4302001
conda activate base      # this passes your conda environment to all the compute nodes
cd /nethome/4302001/backtracking_play/channel/


python3 channel_compute_derivatives.py 