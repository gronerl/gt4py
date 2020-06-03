import os
import numpy as np

import gt4py.storage as gt_store

from tests.reference_cpp_regression import reference_module

from tests.test_integration.utils import generate_test_module

import gt4py.backend as gt_backend
from tests.utils import id_version


def get_reference(test_name, backend, domain, origins, shapes, masks=None):
    reference_data = reference_module.__dict__[test_name](*domain)

    res = {}
    for k, data in reference_data.items():
        if np.isscalar(data):
            res[k] = np.float_(data)
        else:
            try:
                field = gt_store.from_array(
                    data,
                    dtype=np.float_,
                    default_origin=origins[k],
                    shape=shapes[k],
                    backend=backend.name,
                )
            except KeyError:
                field = gt_store.from_array(
                    data,
                    dtype=np.float_,
                    default_origin=origins[k[: -len("_reference")]],
                    shape=shapes[k[: -len("_reference")]],
                    backend=backend.name,
                )

            res[k] = field
    return res


def run_horizontal_diffusion(backend, domain, ntrials):
    backend = gt_backend.from_name(backend)

    origins = {"in_field": (2, 2, 0), "out_field": (0, 0, 0), "coeff": (0, 0, 0)}
    shapes = {k: tuple(domain[i] + 2 * origins[k][i] for i in range(3)) for k in origins.keys()}
    name = "horizontal_diffusion"

    arg_fields = get_reference(name, backend, domain, origins, shapes)
    arg_fields = {k: v for k, v in arg_fields.items() if not k.endswith("_reference")}

    testmodule = generate_test_module(
        "horizontal_diffusion", backend, id_version=id_version, rebuild=False
    )
    for k in arg_fields:
        if hasattr(arg_fields[k], "host_to_device"):
            arg_fields[k].host_to_device()
    exec_infos = []
    for i in range(ntrials):
        exec_info = {}
        testmodule(
            **arg_fields,
            # **{k: v.view(np.ndarray) for k, v in arg_fields.items()},
            domain=domain,
            # origin={"_all_": (0, 0, 0)},
            origin=origins,
            # _origin_={
            #    k: [oo[0] if isinstance(oo, tuple) else oo for oo in o] for k, o in origins.items()
            # },
            exec_info=exec_info,
        )
        exec_infos.append(exec_info)

    return exec_infos


def run_vertical_advection_dycore(backend, domain, ntrials):
    backend = gt_backend.from_name(backend)

    origins = {
        "utens_stage": (0, 0, 0),
        "u_stage": (0, 0, 0),
        "wcon": (0, 0, 0),
        "u_pos": (0, 0, 0),
        "utens": (0, 0, 0),
    }
    shapes = {
        "utens_stage": domain,
        "u_stage": domain,
        "wcon": tuple(d + 1 if i == 0 else d for i, d in enumerate(domain)),
        "u_pos": domain,
        "utens": domain,
    }
    name = "vertical_advection_dycore"

    arg_fields = get_reference(name, backend, domain, origins, shapes)
    arg_fields = {k: v for k, v in arg_fields.items() if not k.endswith("_reference")}
    testmodule = generate_test_module(
        "vertical_advection_dycore", backend, id_version=id_version, rebuild=False
    )
    for k in arg_fields:
        if hasattr(arg_fields[k], "host_to_device"):
            arg_fields[k].host_to_device()
    exec_infos = []
    for i in range(ntrials):
        exec_info = {}
        testmodule.run(
            **arg_fields,
            # **{k: v.view(np.ndarray) for k, v in arg_fields.items()},
            _domain_=domain,
            _origin_=origins,
            # _origin_={
            #    k: [oo[0] if isinstance(oo, tuple) else oo for oo in o] for k, o in origins.items()
            # },
            exec_info=exec_info,
        )
        exec_infos.append(exec_info)

    return exec_infos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("nxy", type=int)
    parser.add_argument("ntrials", type=int)
    parser.add_argument("stencil_name", type=str)
    parser.add_argument("backend", type=str)
    parser.add_argument("outfile", type=str)

    args = parser.parse_args()
    nxy = args.nxy
    ntrials = args.ntrials
    stencil_name = args.stencil_name
    backend = args.backend
    nz = 80

    print(nxy, ntrials, stencil_name, backend)

    if stencil_name == "horizontal_diffusion":
        test_function = run_horizontal_diffusion
    elif stencil_name == "vertical_advection":
        test_function = run_vertical_advection_dycore
    else:
        assert False
    exec_infos = run_horizontal_diffusion(backend, (nxy, nxy, nz), ntrials)
    assert len(exec_infos) == ntrials
    keys = [
        "call_start_time",
        "call_run_start_time",
        "run_start_time",
        "start_run_cpp_time",
        "end_run_cpp_time",
        "run_end_time",
        "call_run_end_time",
        "call_end_time",
    ]
    if not os.path.exists(args.outfile):
        with open(args.outfile, "w") as handle:
            handle.write("stencil_name; backend;nxy; nz;")
            for key in keys:
                handle.write(key + ";")
            handle.write("\n")
    with open(args.outfile, "a") as handle:
        for exec_info in exec_infos:
            handle.write(f"{stencil_name};{backend};{nxy};{nz};")
            for key in keys:
                handle.write(str(exec_info.get(key, 0.0)) + ";")
            handle.write("\n")
