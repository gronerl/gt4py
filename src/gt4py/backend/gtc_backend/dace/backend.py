# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import re
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

import dace
import numpy as np

import gt4py.definitions
from eve import codegen
from eve.codegen import MakoTemplate as as_mako
from gt4py import backend as gt_backend
from gt4py import gt_src_manager
from gt4py.backend import BaseGTBackend, CLIBackendMixin, make_args_data_from_gtir
from gt4py.backend.gt_backends import make_x86_layout_map
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.backend.module_generator import compute_legacy_extents
from gt4py.ir import StencilDefinition
from gtc import gtir, gtir_to_oir
from gtc.dace.oir_to_dace import OirSDFGBuilder
from gtc.dace.utils import array_dimensions
from gtc.passes.gtir_dtype_resolver import resolve_dtype
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters
from gtc.passes.gtir_upcaster import upcast
from gtc.passes.oir_dace_optimizations import GraphMerging, optimize_horizontal_executions
from gtc.passes.oir_optimizations.caches import (
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)
from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging, OnTheFlyMerging
from gtc.passes.oir_optimizations.pruning import NoFieldAccessPruning
from gtc.passes.oir_optimizations.temporaries import (
    LocalTemporariesToScalars,
    WriteBeforeReadTemporariesToScalars,
)
from gtc.passes.oir_optimizations.vertical_loop_merging import AdjacentLoopMerging


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


class GTCDaCeExtGenerator:
    def __init__(self, class_name, module_name, gt_backend_t, builder):
        self.class_name = class_name
        self.module_name = module_name
        self.gt_backend_t = gt_backend_t
        self.builder = builder

    def __call__(self, definition_ir: StencilDefinition) -> Dict[str, Dict[str, str]]:
        gtir = DefIRToGTIR.apply(definition_ir)
        gtir_without_unused_params = prune_unused_parameters(gtir)
        dtype_deduced = resolve_dtype(gtir_without_unused_params)
        upcasted = upcast(dtype_deduced)
        oir = gtir_to_oir.GTIRToOIR().visit(upcasted)
        oir = self._optimize_oir(oir)
        sdfg = OirSDFGBuilder().visit(oir)
        sdfg.expand_library_nodes(recursive=True)

        # TODO uncomment once the branch dace/linus-fixes-8 is merged into dace/master
        # sdfg.apply_strict_transformations(validate=True) # noqa: E800 Found commented out code

        sdfg = self._expand_and_wrap_sdfg(gtir, sdfg)

        for tmp_sdfg in sdfg.all_sdfgs_recursive():
            tmp_sdfg.transformation_hist = []
            tmp_sdfg.orig_sdfg = None
        sdfg.save(
            self.builder.module_path.joinpath(
                os.path.dirname(self.builder.module_path), self.builder.module_name + ".sdfg"
            )
        )

        implementation = DaCeComputationCodegen.apply(gtir, sdfg)
        bindings = DaCeBindingsCodegen.apply(gtir, sdfg, module_name=self.module_name)

        bindings_ext = ".cu" if self.gt_backend_t == "gpu" else ".cpp"
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings" + bindings_ext: bindings},
        }

    def _optimize_oir(self, oir):
        # oir = optimize_horizontal_executions(oir, GraphMerging)
        # oir = GreedyMerging().visit(oir)
        # oir = AdjacentLoopMerging().visit(oir)
        # oir = LocalTemporariesToScalars().visit(oir)
        # oir = WriteBeforeReadTemporariesToScalars().visit(oir)
        # oir = OnTheFlyMerging().visit(oir)
        # oir = NoFieldAccessPruning().visit(oir)
        # oir = IJCacheDetection().visit(oir)
        # oir = KCacheDetection().visit(oir)
        # oir = PruneKCacheFills().visit(oir)
        # oir = PruneKCacheFlushes().visit(oir)
        return oir

    def _expand_and_wrap_sdfg(self, gtir: gtir.Stencil, inner_sdfg: dace.SDFG) -> dace.SDFG:
        wrapper_sdfg = dace.SDFG(inner_sdfg.name + "_offset_wrapper")
        wrapper_state = wrapper_sdfg.add_state(inner_sdfg.name + "_offset_wrapper_state")

        args_data = make_args_data_from_gtir(GtirPipeline(gtir))

        # stencils without effect
        if all(info is None for info in args_data.field_info.values()):
            return wrapper_sdfg

        inner_sdfg.expand_library_nodes(recursive=True)
        extents = compute_legacy_extents(gtir, allow_negative=True)

        inputs = {
            name
            for name, info in args_data.field_info.items()
            if info is not None and info.access != gt4py.definitions.AccessKind.WRITE
        }
        outputs = {
            name
            for name, info in args_data.field_info.items()
            if info is not None and info.access != gt4py.definitions.AccessKind.READ
        }

        nsdfg = wrapper_state.add_nested_sdfg(inner_sdfg, None, inputs=inputs, outputs=outputs)

        subset_strs = {}
        for name, info in args_data.field_info.items():
            if info is None:
                continue

            extent = [
                e for e, a in zip(extents[name], "IJK") if a in args_data.field_info[name].axes
            ]
            shape = [
                s + abs(max(el, 0)) for s, (el, eh) in zip(inner_sdfg.arrays[name].shape, extent)
            ] + [str(d) for d in args_data.field_info[name].data_dims]
            wrapper_sdfg.add_array(
                name,
                strides=inner_sdfg.arrays[name].strides,
                shape=shape,
                dtype=inner_sdfg.arrays[name].dtype,
            )

            subset_strs[name] = ",".join(
                [
                    f"{max(e[0], 0)}:{max(e[0], 0)+s}"
                    for e, s in zip(extent, inner_sdfg.arrays[name].shape)
                ]
                + [f"0:{d}" for d in args_data.field_info[name].data_dims]
            )
        for name in inputs:
            wrapper_state.add_edge(
                wrapper_state.add_read(name),
                None,
                nsdfg,
                name,
                dace.Memlet.simple(name, subset_str=subset_strs[name]),
            )
        for name in outputs:
            wrapper_state.add_edge(
                nsdfg,
                name,
                wrapper_state.add_write(name),
                None,
                dace.Memlet.simple(name, subset_str=subset_strs[name]),
            )

        symbol_mapping = {sym: sym for sym in inner_sdfg.symbols.keys()}
        # transients = set()
        for name, array in inner_sdfg.arrays.items():
            if array.transient:
                # transients.add(name)
                stride = 1
                for symbol, size in zip(reversed(array.strides), reversed(array.shape)):
                    symbol_mapping[str(symbol)] = stride
                    stride *= size

        for old, new in symbol_mapping.items():
            wrapper_sdfg.replace(old, new)

        for name, info in args_data.parameter_info.items():
            if info is not None and name not in wrapper_sdfg.symbols:
                wrapper_sdfg.add_symbol(name, nsdfg.sdfg.symbols[name])

        #     for name, info in args_data.field_info.items():
        #         assert info is None
        #         wrapper_sdfg.add_array()
        #         wrapper_state.add_edge(
        #             wrapper_state.add_read(name),
        #             None,
        #             nsdfg,
        #             name,
        #             dace.Memlet(),
        #         )
        #         wrapper_state.add_edge(
        #             nsdfg,
        #             name,
        #             wrapper_state.add_read(name),
        #             None,
        #             dace.Memlet(),
        #         )

        # for sdfg, name, array in inner_sdfg.arrays_recursive():
        #     if name not in transients:
        #         continue
        #     strides = [*array.strides]
        #     for i, stride in enumerate(array.strides):
        #         if str(stride) in symbol_mapping:
        #             if str(stride) in sdfg.symbols and re.match(f"__.*_._stride", str(stride)):
        #                 sdfg.remove_symbol(str(stride))
        #
        #             strides[i] = dace.symbolic.pystr_to_symbolic(symbol_mapping[str(stride)])
        #             strides[i].free_symbols
        #             for sym in strides[i].free_symbols:
        #                 if str(sym) not in sdfg.symbols:
        #                     sdfg.add_symbol(str(sym), inner_sdfg.symbols[str(sym)])
        #     array.strides = tuple(strides)

        return wrapper_sdfg


class DaCeComputationCodegen:

    template = as_mako(
        """
        auto ${name}(const std::array<gt::uint_t, 3>& domain) {
            return [domain](${",".join(functor_args)}) {
                const int __I = domain[0];
                const int __J = domain[1];
                const int __K = domain[2];
                ${name}_t dace_handle;
                auto allocator = gt::sid::make_cached_allocator(&std::make_unique<char[]>);
                ${"\\n".join(tmp_allocs)}
                __program_${name}(${",".join(["&dace_handle", *dace_args])});
            };
        }
        """
    )

    def generate_tmp_allocs(self, sdfg):
        fmt = "dace_handle.{name} = allocate(allocator, gt::meta::lazy::id<{dtype}>(), {size})();"
        return [
            fmt.format(name=name, dtype=array.dtype.ctype, size=array.total_size)
            for sdfg, name, array in sdfg.arrays_recursive()
            if array.transient and array.lifetime == dace.AllocationLifetime.Persistent
        ]

    @classmethod
    def apply(cls, gtir, sdfg: dace.SDFG):
        self = cls()

        code_objects = dace.SDFG.from_json(sdfg.to_json()).generate_code()
        computations = code_objects[[co.title for co in code_objects].index("Frame")].clean_code
        lines = computations.split("\n")
        computations = "\n".join(lines[0:2] + lines[3:])  # remove import of not generated file
        computations = codegen.format_source("cpp", computations, style="LLVM")
        interface = cls.template.definition.render(
            name=sdfg.name,
            dace_args=self.generate_dace_args(gtir, sdfg),
            functor_args=self.generate_functor_args(sdfg),
            tmp_allocs=self.generate_tmp_allocs(sdfg),
        )
        generated_code = f"""#include <gridtools/sid/sid_shift_origin.hpp>
                             #include <gridtools/sid/allocator.hpp>
                             #include <gridtools/stencil/cartesian.hpp>
                             namespace gt = gridtools;
                             {computations}
                             {interface}
                             """
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

    def __init__(self):
        self._unique_index = 0

    def generate_dace_args(self, gtir, sdfg):
        offset_dict: Dict[str, Tuple[int, int, int]] = {
            k: (-v[0][0], -v[1][0], -v[2][0]) for k, v in compute_legacy_extents(gtir).items()
        }
        symbols = {f"__{var}": f"__{var}" for var in "IJK"}
        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
                # symbols[f"__{name}_K_stride"] = "1"
                # symbols[f"__{name}_J_stride"] = str(array.shape[2])
                # symbols[f"__{name}_I_stride"] = str(array.shape[1] * array.shape[2])
            else:
                dims = [dim for dim, select in zip("IJK", array_dimensions(array)) if select]
                data_ndim = len(array.shape) - len(dims)

                # api field strides
                fmt = "gt::sid::get_stride<{dim}>(gt::sid::get_strides(__{name}_sid))"

                symbols.update(
                    {
                        f"__{name}_{dim}_stride": fmt.format(
                            dim=f"gt::stencil::dim::{dim.lower()}", name=name
                        )
                        for dim in dims
                    }
                )
                symbols.update(
                    {
                        f"__{name}_d{dim}_stride": fmt.format(
                            dim=f"gt::integral_constant<int, {3 + dim}>", name=name
                        )
                        for dim in range(data_ndim)
                    }
                )

                # api field pointers
                fmt = """gt::sid::multi_shifted(
                             gt::sid::get_origin(__{name}_sid)(),
                             gt::sid::get_strides(__{name}_sid),
                             std::array<gt::int_t, {ndim}>{{{origin}}}
                         )"""
                origin = tuple(
                    -offset_dict[name][idx]
                    for idx, var in enumerate("IJK")
                    if any(
                        dace.symbolic.pystr_to_symbolic(f"__{var}") in s.free_symbols
                        for s in array.shape
                        if hasattr(s, "free_symbols")
                    )
                )
                symbols[name] = fmt.format(
                    name=name, ndim=len(array.shape), origin=",".join(str(o) for o in origin)
                )
        # the remaining arguments are variables and can be passed by name
        for sym in sdfg.signature_arglist(with_types=False, for_call=True):
            if sym not in symbols:
                symbols[sym] = sym

        # return strings in order of sdfg signature
        return [symbols[s] for s in sdfg.signature_arglist(with_types=False, for_call=True)]

    def generate_functor_args(self, sdfg: dace.SDFG):
        res = []
        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
            res.append(f"auto && __{name}_sid")
        for name, dtype in ((n, d) for n, d in sdfg.symbols.items() if not n.startswith("__")):
            res.append(dtype.as_arg(name))
        return res


class DaCeBindingsCodegen:
    def __init__(self):
        self._unique_index: int = 0

    def unique_index(self) -> int:
        self._unique_index += 1
        return self._unique_index

    mako_template = as_mako(
        """#include <chrono>
           #include <pybind11/pybind11.h>
           #include <pybind11/stl.h>
           #include <gridtools/storage/adapter/python_sid_adapter.hpp>
           #include <gridtools/stencil/cartesian.hpp>
           #include <gridtools/stencil/global_parameter.hpp>
           #include <gridtools/sid/sid_shift_origin.hpp>
           #include <gridtools/sid/rename_dimensions.hpp>
           #include "computation.hpp"
           namespace gt = gridtools;
           namespace py = ::pybind11;
           PYBIND11_MODULE(${module_name}, m) {
               m.def("run_computation", [](
               ${','.join(["std::array<gt::uint_t, 3> domain", *entry_params, 'py::object exec_info'])}
               ){
                   if (!exec_info.is(py::none()))
                   {
                       auto exec_info_dict = exec_info.cast<py::dict>();
                       exec_info_dict["run_cpp_start_time"] = static_cast<double>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::high_resolution_clock::now().time_since_epoch()).count())/1e9;
                   }

                   ${name}(domain)(${','.join(sid_params)});

                   if (!exec_info.is(py::none()))
                   {
                       auto exec_info_dict = exec_info.cast<py::dict>();
                       exec_info_dict["run_cpp_end_time"] = static_cast<double>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::high_resolution_clock::now().time_since_epoch()).count()/1e9);
                   }

               }, "Runs the given computation");}
        """
    )

    def generate_entry_params(self, gtir: gtir.Stencil, sdfg: dace.SDFG):
        res = {}
        import dace.data

        for name in sdfg.signature_arglist(with_types=False, for_call=True):
            if name in sdfg.arrays:
                data = sdfg.arrays[name]
                assert isinstance(data, dace.data.Array)
                res[name] = "py::buffer {name}, std::array<gt::uint_t,{ndim}> {name}_origin".format(
                    name=name,
                    ndim=len(data.shape),
                )
            elif name in sdfg.symbols and not name.startswith("__"):
                assert name in sdfg.symbols
                res[name] = "{dtype} {name}".format(dtype=sdfg.symbols[name].ctype, name=name)
        return list(res[node.name] for node in gtir.params if node.name in res)

    def generate_sid_params(self, sdfg: dace.SDFG):
        res = []
        import dace.data

        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
            dimensions = array_dimensions(array)
            domain_ndim = sum(dimensions)
            data_ndim = len(array.shape) - domain_ndim
            sid_def = """gt::as_{sid_type}<{dtype}, {num_dims},
                gt::integral_constant<int, {unique_index}>>({name})""".format(
                sid_type="cuda_sid"
                if array.storage in [dace.StorageType.GPU_Global, dace.StorageType.GPU_Shared]
                else "sid",
                name=name,
                dtype=array.dtype.ctype,
                unique_index=self.unique_index(),
                num_dims=len(array.shape),
            )
            sid_def = "gt::sid::shift_sid_origin({sid_def}, {name}_origin)".format(
                sid_def=sid_def, name=name
            )

            if domain_ndim != 3:
                gt_dims = [
                    f"gt::stencil::dim::{dim}"
                    for dim in "ijk"
                    if any(
                        dace.symbolic.pystr_to_symbolic(f"__{dim.upper()}") in s.free_symbols
                        for s in array.shape
                        if hasattr(s, "free_symbols")
                    )
                ]
                if data_ndim:
                    gt_dims += [
                        f"gt::integral_constant<int, {3 + dim}>" for dim in range(data_ndim)
                    ]
                sid_def = "gt::sid::rename_numbered_dimensions<{gt_dims}>({sid_def})".format(
                    gt_dims=", ".join(gt_dims), sid_def=sid_def
                )

            res.append(sid_def)
        # pass scalar parameters as variables
        for name in (n for n in sdfg.symbols.keys() if not n.startswith("__")):
            res.append(name)
        return res

    def generate_sdfg_bindings(self, gtir, sdfg, module_name):

        return self.mako_template.render_values(
            name=sdfg.name,
            module_name=module_name,
            entry_params=self.generate_entry_params(gtir, sdfg),
            sid_params=self.generate_sid_params(sdfg),
        )

    @classmethod
    def apply(cls, gtir: gtir.Stencil, sdfg: dace.SDFG, module_name: str) -> str:
        generated_code = cls().generate_sdfg_bindings(gtir, sdfg, module_name=module_name)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


class DaCePyExtModuleGenerator(gt_backend.PyExtModuleGenerator):
    def generate_imports(self):
        res = super().generate_imports()
        return res + "\nimport dace\nimport copy"

    def generate_module_members(self) -> str:
        res = super().generate_module_members()

    def generate_class_members(self):
        res = super().generate_class_members()
        filepath = self.builder.module_path.joinpath(
            os.path.dirname(self.builder.module_path), self.builder.module_name + ".sdfg"
        )
        res += """
_sdfg = None

def __new__(cls, *args, **kwargs):
    res = super().__new__(cls, *args, **kwargs)
    cls._sdfg = dace.SDFG.from_file('{filepath}')
    return res

@property
def sdfg(self) -> dace.SDFG: 
    return copy.deepcopy(self._sdfg)


""".format(
            filepath=filepath
        )
        return res


@gt_backend.register
class GTCDaceBackend(BaseGTBackend, CLIBackendMixin):
    """DaCe python backend using gtc."""

    name = "gtc:dace"
    GT_BACKEND_T = "dace"
    languages = {"computation": "c++", "bindings": ["python"]}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": make_x86_layout_map,
        "is_compatible_layout": lambda x: True,
        "is_compatible_type": lambda x: isinstance(x, np.ndarray),
    }

    MODULE_GENERATOR_CLASS = DaCePyExtModuleGenerator

    options = BaseGTBackend.GT_BACKEND_OPTS
    PYEXT_GENERATOR_CLASS = GTCDaCeExtGenerator  # type: ignore
    USE_LEGACY_TOOLCHAIN = False

    def generate_extension(self) -> Tuple[str, str]:
        return self.make_extension(gt_version=2, ir=self.builder.definition_ir, uses_cuda=False)

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources(2) and not gt_src_manager.install_gt_sources(2):
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name: Optional[str]
        pyext_file_path: Optional[str]

        # TODO(havogt) add bypass if computation has no effect
        pyext_module_name, pyext_file_path = self.generate_extension()

        # Generate and return the Python wrapper class
        return self.make_module(
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )
