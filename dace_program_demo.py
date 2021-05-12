import dace
import numpy as np

from gt4py import gtscript
from gt4py.gtscript import Field


@gtscript.as_sdfg
def simple_horizontal_diffusion(
    in_field: Field[np.float32], coeff: Field[np.float32], out_field: Field[np.float32]
):
    with computation(PARALLEL), interval(...):
        lap_field = 4.0 * in_field[0, 0, 0] - (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
        )
        flx_field = lap_field[1, 0, 0] - lap_field[0, 0, 0]
        fly_field = lap_field[0, 1, 0] - lap_field[0, 0, 0]
        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0]
        )


I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


@dace.program
def call_hdiff(
    in_field: dace.float32[I + 4, J + 4, K],
    coeff: dace.float32[I, J, K],
    out_field: dace.float32[I, J, K],
):
    for _ in range(100):
        simple_horizontal_diffusion(in_field=in_field, coeff=coeff, out_field=out_field)


if __name__ == "__main__":
    in_field = np.ones((14, 15, 12), dtype=np.float32)
    coeff = np.random.randn(10, 11, 12).astype(np.float32)
    out_field = np.zeros((10, 11, 12), dtype=np.float32)
    sdfg = simple_horizontal_diffusion
    from dace.frontend.python.parser import DaceProgram

    call_hdiff: DaceProgram
    sdfg2 = call_hdiff.to_sdfg()
    sdfg2.save("tmp.sdfg")
    call_hdiff(in_field, coeff, out_field)

# # sdfg = dace.SDFG('outer')
# # state = sdfg.add_state('outer_state')
# #
# # inner_sdfg = dace.SDFG('inner')
# # inner_state = inner_sdfg.add_state('inner_state')
# # inner_sdfg.add_symbol('value', stype=dace.dtypes.int64)
# #
# # nsdfg = state.add_nested_sdfg(inner_sdfg,parent=None, inputs = {'value'}, outputs={})
# # sdfg.add_array('data', shape=(10,), dtype=dace.dtypes.int64)
# #
# # state.add_edge(state.add_read('data'), None, nsdfg, 'value', dace.Memlet.simple('data', subset_str='0'))
# # sdfg.validate()
# #
# # sdfg.save('tmp.sdfg')
# print('ahoj')
# a,b = 1,2,3
#
# W = dace.symbol('W')
# H = dace.symbol('H')
# D = dace.symbol('D', dtype=dace.int64)
# @dace.program(dace.float32[H, W], dace.float32[W, H])
# def mytranspose(A, B):
#
#     tmpp = np.empty((H,W,2), dtype=np.float32)
#     d1,d2 = A.strides
#     asdf = [1,2,3]
#     # print(asdf)
#     for k in range(2):
#         for i in range(H):
#             for j in range(W):
#                 tmpp[j, i, k] = A[i, j]
#     B[...] = tmpp[:, :, d1]
#
# tmp = mytranspose.to_sdfg()
# tmp.save('tmp.sdfg')
# #
# #
# # @dace.program(dace.float32[H, W], dace.float32[W, H])
# # def transpose(A, B):
# #     c = 1.0
# #     mytranspose(A, B, D=c)
# #
# # if __name__ == "__main__":
# #
# #     W.set(10)
# #     H.set(5)
# #
# #     print('Transpose %dx%d' % (W.get(), H.get()))
# #
# #     A = np.random.rand(H.get(), W.get()).astype(np.float32)
# #     B = np.zeros([W.get(), H.get()], dtype=np.float32)
# #     sdfg: dace.SDFG = mytranspose.to_sdfg()
# #     sdfg.save('inner.sdfg')
# #     print(sdfg.signature_arglist(for_call=True, with_arrays=True, with_types=True))
# #     sdfg = transpose.to_sdfg()
# #     sdfg.save('tmp.sdfg')
# #     transpose(A, B)
# #
# #     if dace.Config.get_bool('profiling'):
# #         dace.timethis('transpose', 'numpy', (H.get() * W.get()), np.transpose,
# #                       A)
# #     diff = np.linalg.norm(np.transpose(A) - B) / (H.get() * W.get())
# #     print("Difference:", diff)
# #     print("==== Program end ====")
# #     exit(0 if diff <= 1e-5 else 1)
