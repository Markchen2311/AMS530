from mpi4py import MPI
import numpy as np
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def initialize_grid(n, boundary_value=0):
    u = np.zeros((n, n))
    # Set boundary conditions
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = boundary_value
    return u

def jacobi_step_parallel(u, f, h, rank, size):
    u_new = np.copy(u)
    n = u.shape[0]

    # Exchange boundaries with neighboring processes
    if rank > 0:
        comm.Sendrecv(u[1, :], dest=rank-1, sendtag=11,
                      recvbuf=u[0, :], source=rank-1, recvtag=12)
    if rank < size - 1:
        comm.Sendrecv(u[-2, :], dest=rank+1, sendtag=12,
                      recvbuf=u[-1, :], source=rank+1, recvtag=11)
    
    for i in range(1, n-1):
        for j in range(1, n-1):
            u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - h**2 * f[i, j])
    return u_new

def gauss_seidel_step_parallel(u, f, h, rank, size):
    n = u.shape[0]

    # Exchange boundaries with neighboring processes
    if rank > 0:
        comm.Sendrecv(u[1, :], dest=rank-1, sendtag=21,
                      recvbuf=u[0, :], source=rank-1, recvtag=22)
    if rank < size - 1:
        comm.Sendrecv(u[-2, :], dest=rank+1, sendtag=22,
                      recvbuf=u[-1, :], source=rank+1, recvtag=21)
    
    for i in range(1, n-1):
        for j in range(1, n-1):
            u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - h**2 * f[i, j])
    return u

def compute_residual(u, f, h):
    n = u.shape[0]
    residual = np.zeros_like(u)
    for i in range(1, n-1):
        for j in range(1, n-1):
            residual[i, j] = f[i, j] - (4 * u[i, j] - u[i+1, j] - u[i-1, j] - u[i, j+1] - u[i, j-1]) / h**2
    return np.max(np.abs(residual))

def solve_poisson_parallel(n, f, method='jacobi', tol=1e-5, max_iter=1000):
    h = 1.0 / (n - 1)
    local_n = n // size
    u = initialize_grid(local_n + 2)  

    # Timing start
    if rank == 0:
        start_time = time.time()

    for k in range(max_iter):
        u_old = np.copy(u)
        
        if method == 'jacobi':
            u = jacobi_step_parallel(u, f, h, rank, size)
        elif method == 'gauss_seidel':
            u = gauss_seidel_step_parallel(u, f, h, rank, size)
        
        # Compute the residual for convergence
        local_residual = compute_residual(u[1:-1, 1:-1], f[1:-1, 1:-1], h)
        residual = comm.allreduce(local_residual, op=MPI.MAX)
        
        if rank == 0:
            print(f"Iteration {k}, Residual: {residual}")
        
        if residual < tol:
            break

    # Timing end
    if rank == 0:
        end_time = time.time()
        print(f"Total time for {method} with {size} processes: {end_time - start_time} seconds")

    # Gather results back on rank 0 for analysis if needed
    u_contiguous = np.ascontiguousarray(u[1:-1, 1:-1])
    u_global = None
    if rank == 0:
        u_global = np.zeros((n, n))
    comm.Gather(u_contiguous, u_global, root=0)
    
    return u_global if rank == 0 else None

# Define parameters for a 128x128 grid
n = 128
f = np.ones((n // size + 2, n))  

# Solve using Jacobi method
if rank == 0:
    print("Starting Jacobi method...")
u_jacobi = solve_poisson_parallel(n, f, method='jacobi')

# Solve using Gauss-Seidel method
if rank == 0:
    print("Starting Gauss-Seidel method...")
u_gauss_seidel = solve_poisson_parallel(n, f, method='gauss_seidel')

if rank == 0:
    print("Solution complete.")
