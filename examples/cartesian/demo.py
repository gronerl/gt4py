import numpy as np
import matplotlib.pyplot as plt

import gt4py.storage
import gt4py.cartesian.gtscript as gtscript

backend = "dace:cpu"  # options: "numpy", "gt:cpu_ifirst", "gt:cpu_kfirst", "gt:gpu", "dace:cpu", "dace:gpu"
dtype = np.float64


@gtscript.stencil(
    backend, rebuild=True
)  # this decorator triggers compilation of the stencil
def diffusion(
    in_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
    coeff: dtype,
):
    with computation(PARALLEL), interval(1, -1):
        lap_field = -6.0 * in_field[0, 0, 0] + (
            in_field[1, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[0, 1, 0]
            + in_field[0, -1, 0]
            + in_field[0, 0, 1]
            + in_field[0, 0, -1]
        )
        out_field[...] = in_field[0, 0, 0] + coeff * lap_field[0, 0, 0]


# Initialize example data
N = 30
shape = (N, N, N)
origin = (1, 1, 0)


# initialize data
in_storage = gt4py.storage.zeros(
    shape=shape, dtype=dtype, backend=backend, aligned_index=origin
)
in_storage[N // 2 - 2 : N // 2 + 2, N // 2 - 2 : N // 2 + 2, :] = 1.0
out_storage = gt4py.storage.zeros(
    shape=shape, dtype=dtype, backend=backend, aligned_index=origin
)

# Plot input
plt.imshow(in_storage[:, :, N // 2])
plt.show()

# call the compiled stencil
diffusion(in_storage, out_storage, coeff=0.25, origin=origin)


# Plot output
plt.imshow(out_storage[:, :, N // 2])
plt.show()


# Using stencils in parsed context
import dace

DaceArrayType = dace.dtype_to_typeclass(dtype)[shape[0], shape[1], shape[2]]


@dace.program  # invoke general-purpose dace parser & compiler
def diffusion_steps(
    in_field: DaceArrayType, out_field: DaceArrayType, coeff: dtype, n: int
):
    tmp = np.copy(in_field)
    for _ in range(n):
        diffusion(tmp, out_field, coeff, orign=origin)
        tmp[...] = out_field


# extract SDFG IR and display it
sdfg = diffusion_steps.to_sdfg()
sdfg.view()
sdfg.expand_library_nodes()
sdfg.view()
