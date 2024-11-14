#!/bin/bash
#SBATCH --job-name=mpi_python_job         # Job name
#SBATCH --output=output.txt               # Output file
#SBATCH --nodes=4                         # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of MPI tasks per node 
#SBATCH --time=05:00                      # Time limit
#SBATCH -p short-40core                   # Partition

# Load necessary modules
module load gcc/7.1.0
module load openmpi4/gcc/4.1.2 
module load python/3.11.2                    # Load Python module 

pip install --user mpi4py

# Run the Python MPI program
mpirun -n 4 python poisson_solver.py    # Run the Python program with MPI