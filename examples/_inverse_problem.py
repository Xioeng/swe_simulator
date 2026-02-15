"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from swe_simulator.result import SWEResult

swe_result = SWEResult().load("_output/result.pkl")

print(swe_result)
import matplotlib.pyplot as plt

plot = True

np.nan_to_num(swe_result.solution, nan=0.0, copy=False)
if plot:
    xx, yy = swe_result.meshgrid_metric
    fig, axes = plt.subplots(2, 2, figsize=(12, 5))
    im1 = axes[0, 0].contourf(xx, yy, swe_result.bathymetry, cmap="viridis")
    axes[0, 0].set_title("Bathymetry")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0, 0])

    # Initial condition subplot
    print(swe_result.solution.shape)
    im2 = axes[0, 1].contourf(xx, yy, swe_result.solution[-1, 0, :, :], cmap="plasma")
    axes[0, 1].set_title("Initial Condition")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[0, 1])

    # Velocity subplot
    im3 = axes[1, 0].contourf(xx, yy, swe_result.solution[-1, 1, :, :], cmap="inferno")
    axes[1, 0].set_title("Velocity")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    plt.colorbar(im3, ax=axes[1, 0])

    # Pressure subplot
    im4 = axes[1, 1].contourf(xx, yy, swe_result.solution[-1, 2, :, :], cmap="cividis")
    axes[1, 1].set_title("Pressure")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Y")
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()
