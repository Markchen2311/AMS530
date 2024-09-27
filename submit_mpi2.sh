#!/bin/bash
#SBATCH --job-name=mpi_cpp_job      
#SBATCH --output=mpi_output.txt   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28                
#SBATCH --time=05:00             
#SBATCH -p short-40core         


module load gcc/9.3.0              
module load openmpi/4.0.3           


mpicxx mpi_program.cpp -o mpi_program


mpirun -np 28 ./mpi_program