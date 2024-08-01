#!/bin/bash -l
#
#SBATCH -J moi_backward_2d_div       #name of your job   
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 5-00:00:00         # time in d-hh:mm:ss you want to reserve for the job
#SBATCH -n 1                # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH --mem=70G          
#SBATCH -o moi_backward_2d_div.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e moi_backward_2d_div.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM
#SBATCH --mail-user=b.j.h.r.reijnders@uu.nl 
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

cd /nethome/4302001
conda activate parcels3      # this passes your conda environment to all the compute nodes
# conda activate parcels      # this passes your conda environment to all the compute nodes

cd /nethome/4302001/backtracking_play/moi/

counter=0
for file in /nethome/4302001/output_data/backtracking/moi/divergence/MOi_divergence_*_forward_init_2015-02-01_T180.zarr; do
    python3 moi_backward_from_forward_2d_divergence.py $file --T_integration 10 30 90 180 &
    counter=$((counter+1))
    
    if (( counter % 3 == 0 )); then
        wait # wait until the current batch of 3 processes finish
    fi
done

wait 


