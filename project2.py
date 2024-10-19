from mpi4py import MPI
import numpy as np
import math

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Heart function to define the surface boundary
def heart_surface(x, y, z):
    return (x**2 + (3/2 * y)**2 + z**2 - 1)**3 - (x**2 + (1/50) * (3/2 * y)**2) * z**3

# Density function for mass calculation at time t = π/6
def density(x, y, z, omega, t):
    rho_0 = 1  # Assume constant initial density
    return rho_0 * (1 + math.sin(omega * t)) / 2

# Monte Carlo for surface area
def estimate_surface_area(num_samples, rank, size):
    # Distribute samples to each process
    local_num_samples = num_samples // size  
    count = 0
    flops = 0  # Floating-point operations counter
    for _ in range(local_num_samples):  
        x, y, z = np.random.uniform(-1.5, 1.5, 3)  # Range adjusted for heart size
        flops += 10  # Approximate number of floating point operations
        if abs(heart_surface(x, y, z)) < 0.01:  # Within a small range to estimate surface
            count += 1
    total_count = comm.reduce(count, op=MPI.SUM, root=0)
    total_flops = comm.reduce(flops, op=MPI.SUM, root=0)
    if rank == 0:
        surface_area = total_count / (num_samples) * (3**3)  # Scaled to volume
        flops_per_core = total_flops / size
        print(f"Estimated Surface Area: {surface_area}")
        print(f"Floating-point operations per core (Surface Area): {flops_per_core}")
    return

# Monte Carlo for heart mass using specific density formula
def estimate_mass(num_samples, omega, t, rank, size):
    local_num_samples = num_samples // size  
    local_mass = 0
    flops = 0  # Floating-point operations counter
    for _ in range(local_num_samples):
        x, y, z = np.random.uniform(-1.5, 1.5, 3)
        if abs(heart_surface(x, y, z)) < 0.01:  # Inside heart volume
            local_mass += density(x, y, z, omega, t)
            flops += 10 + 5  # Approximate number of floating point operations
    total_mass = comm.reduce(local_mass, op=MPI.SUM, root=0)
    total_flops = comm.reduce(flops, op=MPI.SUM, root=0)
    if rank == 0:
        flops_per_core = total_flops / size
        print(f"Estimated Mass at t = {t}: {total_mass}")
        print(f"Floating-point operations per core (Mass): {flops_per_core}")
    return

# Parameters
num_samples = 100000
omega = 2 * np.pi  # Arbitrary heart rate, change as needed
t = np.pi / 6  # Time point for mass calculation

# Compute in parallel
estimate_surface_area(num_samples, rank, size)
estimate_mass(num_samples, omega, t, rank, size)

# Finalize MPI (Not necessary as the script will end here)
MPI.Finalize()
