import numpy as np
from clawpack import pyclaw

outdir = "../_output"
frames = range(1000)  # or range(N) if you have more

solutions = []
for f in frames:
    sol = pyclaw.Solution(
        f, outdir=outdir, file_prefix="claw", file_format="petsc", read_aux=False
    )  # read frame f from outdir
    solutions.append(sol.q.copy())  # (num_eqn, nx, ny)

# Stack into array: (n_frames, num_eqn, nx, ny)
solutions = np.stack(solutions, axis=0)
print(solutions.shape)
np.save("solutions.npy", solutions)  # save to disk for later use
