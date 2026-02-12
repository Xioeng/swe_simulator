"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import os
import sys

import deepxde as dde
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from swe_simulator.result import SWEResult

swe_result = SWEResult().load("_output/result.pkl")

print(swe_result)
import matplotlib.pyplot as plt

plot = True

np.nan_to_num(swe_result.solution, nan=0.0, copy=False)
if plot:
    xx, yy = swe_result.meshgrid
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


kf = dde.Variable(0.05)
D = dde.Variable(1.0)


def pde(t, q):
    h, mx, my = q[:, 0:1], q[:, 1:2], q[:, 2:3]
    # Get bathymetry from the loaded SWE result
    xx, yy = swe_result.meshgrid
    bathymetry = swe_result.bathymetry

    # Extract spatial coordinates
    x, y = t[:, 0:1], t[:, 1:2]

    # Compute gradients for shallow water equations
    dh_t = dde.grad.jacobian(q, t, i=0, j=2)  # ∂h/∂t
    dmx_t = dde.grad.jacobian(q, t, i=1, j=2)  # ∂mx/∂t
    dmy_t = dde.grad.jacobian(q, t, i=2, j=2)  # ∂my/∂t

    dh_x = dde.grad.jacobian(q, t, i=0, j=0)  # ∂h/∂x
    dh_y = dde.grad.jacobian(q, t, i=0, j=1)  # ∂h/∂y

    dmx_x = dde.grad.jacobian(q, t, i=1, j=0)  # ∂mx/∂x
    dmy_y = dde.grad.jacobian(q, t, i=2, j=1)  # ∂my/∂y

    dmx_y = dde.grad.jacobian(q, t, i=1, j=1)  # ∂mx/∂y
    dmy_x = dde.grad.jacobian(q, t, i=2, j=0)  # ∂my/∂x

    # Bathymetry gradients (assuming bathymetry is a function of space)
    # For simplicity, using finite differences or assume given gradients
    db_x = 0.0  # ∂b/∂x - replace with actual bathymetry gradient
    db_y = 0.0  # ∂b/∂y - replace with actual bathymetry gradient

    # Wind stress terms (assuming constant for simplicity)
    tau_x = 0.001  # x-component of wind stress
    tau_y = 0.001  # y-component of wind stress

    g = 9.81  # gravitational acceleration

    # Shallow water equations residuals
    # Continuity equation: ∂h/∂t + ∂mx/∂x + ∂my/∂y = 0
    eq_continuity = dh_t + dmx_x + dmy_y

    # x-momentum equation: ∂mx/∂t + ∂/∂x(mx²/h) + ∂/∂y(mx*my/h) + gh∂h/∂x + ghb∂b/∂x = τx
    eq_momentum_x = dmx_t + g * h * dh_x + g * h * db_x - tau_x

    # y-momentum equation: ∂my/∂t + ∂/∂x(mx*my/h) + ∂/∂y(my²/h) + gh∂h/∂y + ghb∂b/∂y = τy
    eq_momentum_y = dmy_t + g * h * dh_y + g * h * db_y - tau_y

    return [eq_continuity, eq_momentum_x, eq_momentum_y]


def fun_bc(x):
    return 0.0


def fun_init(x):
    return np.exp(-20 * x[:, 0:1])


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 10)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_a = dde.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=0
)
bc_b = dde.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=1
)
ic1 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=0)
ic2 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=1)

observe_x, Ca, Cb = gen_traindata()
observe_y1 = dde.icbc.PointSetBC(observe_x, Ca, component=0)
observe_y2 = dde.icbc.PointSetBC(observe_x, Cb, component=1)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_a, bc_b, ic1, ic2, observe_y1, observe_y2],
    num_domain=200,
    num_boundary=100,
    num_initial=100,
    anchors=observe_x,
    num_test=500,
)
net = dde.nn.FNN([2] + [20] * 3 + [2], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[kf, D])
variable = dde.callbacks.VariableValue([kf, D], period=1, filename="variables.dat")
losshistory, train_state = model.train(iterations=80, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
