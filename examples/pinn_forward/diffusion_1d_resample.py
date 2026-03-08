"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np
# Backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
# Backend pytorch
# import torch
# Backend jax
# import jax.numpy as jnp
# Backend paddle
# import paddle

lam=0.01
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    return dy_t - dy_xx + lam * y
    # Backend pytorch
    # return (
    #     dy_t
    #     - dy_xx
    #     + torch.exp(-x[:, 1:])
    #     * (torch.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * x[:, 0:1]))
    # )
    # Backend jax
    # return (
    #     dy_t
    #     - dy_xx
    #     + jnp.exp(-x[:, 1:])
    #     * (jnp.sin(np.pi * x[..., 0:1]) - np.pi ** 2 * jnp.sin(np.pi * x[..., 0:1]))
    # )
    # Backend paddle
    # return (
    #     dy_t
    #     - dy_xx
    #     + paddle.exp(-x[:, 1:])
    #     * (paddle.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * paddle.sin(np.pi * x[:, 0:1]))
    # )


def func(x):
    space = x[:, 0:1]
    time = x[:, 1:2]
    return np.exp(-50 * (space - 0.5) ** 2) * np.exp(-lam * time)


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(
    geomtime,
    lambda x: 0,
    lambda _, on_boundary: on_boundary,
)
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.exp(-50 * (x[:, 0:1] - 0.5) ** 2),
    lambda _, on_initial: on_initial,
)
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=200,
    num_boundary=40,
    num_initial=40,
    num_test=1000,
)
layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=10000)

resampler = dde.callbacks.PDEPointResampler(period=100)


dde.saveplot(losshistory, train_state, issave=True, isplot=True)

import matplotlib.pyplot as plt

x = np.linspace(0, 1, 200)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x, t)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

Y_pred = model.predict(XT)
Y_pred = Y_pred.reshape(len(t), len(x))

plt.figure(figsize=(8, 5))
plt.imshow(Y_pred, extent=[0, 1, 0, 1], origin="lower", aspect="auto")
plt.colorbar(label="Tracer concentration")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Predicted tracer diffusion + decay")
plt.show()
