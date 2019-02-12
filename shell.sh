#!/bin/bash
#SBATCH --job-name="MPI_openMP"
#SBATCH -o log.txt
#SBATCH -p ibm_large

mpirun -np 4  ./main 1500 1200
