#!/bin/bash -l
#
#SBATCH -J moi_backward_3d       #name of your job   
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 5-00:00:00         # time in d-hh:mm:ss you want to reserve for the job
#SBATCH -n 1                # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH --mem=70G          
#SBATCH -o moi_backward_3d.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e moi_backward_3d.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM
#SBATCH --mail-user=b.j.h.r.reijnders@uu.nl 
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

cd /nethome/4302001
conda activate parcels3      # this passes your conda environment to all the compute nodes
# conda activate parcels      # this passes your conda environment to all the compute nodes

cd /nethome/4302001/backtracking_play/moi/atlantic/

python3 moi_run_divergence_Atlantic.py -init_time=20150731 -input='/nethome/4302001/backtracking_play/moi/atlantic/Atlantic_regular_res12_surface_lonlat-dict.pkl' --dt=-600 --advection_mode='3D'
python3 moi_run_divergence_Atlantic.py -input='/nethome/4302001/output_data/backtracking/moi/divergence/MOi_divergence_3D_regular_res12_surface_original_backward_terminus_2015-07-31_T180_dt-600.zarr' --dt=600 --advection_mode='3D' --T_integration 180 90 30 10


