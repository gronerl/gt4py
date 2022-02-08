# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import dace
import dace.data
import numpy as np
from pydantic import validator

import eve
import gtc.oir as oir
from eve import NodeVisitor
from eve.iterators import TraversalOrder, iter_tree
from gt4py.definitions import Extent
from gtc import common
from gtc.common import CartesianOffset, ExprKind, LevelMarker, data_type_to_typestr
from gtc.passes.oir_optimizations.utils import AccessCollector


if TYPE_CHECKING:
    from gtc import daceir as dcir


def axes_list_from_flags(flags):
    from gtc import daceir as dcir

    return [ax for f, ax in zip(flags, dcir.Axis.dims_3d()) if f]


def data_type_to_dace_typeclass(data_type):
    dtype = np.dtype(data_type_to_typestr(data_type))
    return dace.dtypes.typeclass(dtype.type)


def internal_symbols(sdfg: dace.SDFG):
    res = ["__I", "__J", "__K"]
    for name, array in sdfg.arrays.items():
        if isinstance(array, dace.data.Array):
            dimensions = array_dimensions(array)
            n_data_dims = len(array.shape) - sum(dimensions)
            res.extend(
                [f"__{name}_{var}_stride" for idx, var in enumerate("IJK") if dimensions[idx]]
            )
            res.extend([f"__{name}_d{dim}_stride" for dim in range(n_data_dims)])
    return res


def dace_dtype_to_typestr(dtype: Any):
    if not isinstance(dtype, dace.typeclass):
        dtype = dace.typeclass(dtype)
    return dtype.as_numpy_dtype().str


def array_dimensions(array: dace.data.Array):
    dims = [
        any(
            re.match(f"__.*_{k}_stride", str(sym))
            for st in array.strides
            for sym in dace.symbolic.pystr_to_symbolic(st).free_symbols
        )
        or any(
            re.match(f"__{k}", str(sym))
            for sh in array.shape
            for sym in dace.symbolic.pystr_to_symbolic(sh).free_symbols
        )
        for k in "IJK"
    ]
    return tuple(dims)


def get_tasklet_symbol(name, offset, is_target):
    if is_target:
        return f"__{name}"

    acc_name = name + "__"
    suffix = "_".join(
        var + ("m" if o < 0 else "p") + f"{abs(o):d}"
        for var, o in zip("ijk", offset)
        if o is not None and o != 0
    )
    if suffix != "":
        acc_name += suffix
    return acc_name


def get_axis_bound_str(axis_bound, var_name):
    from gtc.common import LevelMarker

    if axis_bound.level == LevelMarker.END:
        return f"{var_name}{axis_bound.offset:+d}"
    else:
        return f"{axis_bound.offset}"


def get_interval_range_str(interval, var_name):
    return "{}:{}".format(
        get_axis_bound_str(interval.start, var_name), get_axis_bound_str(interval.end, var_name)
    )


def get_axis_bound_diff_str(axis_bound1, axis_bound2, var_name: str):

    if axis_bound1 <= axis_bound2:
        axis_bound1, axis_bound2 = axis_bound2, axis_bound1
        sign = "-"
    else:
        sign = ""

    if axis_bound1.level != axis_bound2.level:
        var = var_name
    else:
        var = ""
    return f"{sign}({var}{axis_bound1.offset-axis_bound2.offset:+d})"


def get_interval_length_str(interval, var_name):

    return "({})-({})".format(
        get_axis_bound_str(interval.end, var_name), get_axis_bound_str(interval.start, var_name)
    )


class OIRFieldRenamer(eve.NodeTranslator):
    def visit_FieldAccess(self, node: oir.FieldAccess):
        if node.name not in self._field_table:
            return node
        return oir.FieldAccess(
            name=self._field_table[node.name],
            offset=node.offset,
            dtype=node.dtype,
            data_index=node.data_index,
        )

    def visit_ScalarAccess(self, node: oir.ScalarAccess):
        if node.name not in self._field_table:
            return node
        return oir.ScalarAccess(name=self._field_table[node.name], dtype=node.dtype)

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution):
        assert all(
            decl.name not in self._field_table or self._field_table[decl.name] == decl.name
            for decl in node.declarations
        )
        return node

    def __init__(self, field_table):
        self._field_table = field_table


class CartesianIterationSpace(oir.LocNode):
    i_interval: oir.Interval
    j_interval: oir.Interval

    @validator("i_interval", "j_interval")
    def minimum_domain(cls, v: oir.Interval) -> oir.Interval:
        if (
            v.start.level != LevelMarker.START
            or v.start.offset > 0
            or v.end.level != LevelMarker.END
            or v.end.offset < 0
        ):
            raise ValueError("iteration space must include the whole domain")
        return v

    @staticmethod
    def domain() -> "CartesianIterationSpace":
        return CartesianIterationSpace(
            i_interval=oir.Interval(start=oir.AxisBound.start(), end=oir.AxisBound.end()),
            j_interval=oir.Interval(start=oir.AxisBound.start(), end=oir.AxisBound.end()),
        )

    @staticmethod
    def from_offset(offset: CartesianOffset) -> "CartesianIterationSpace":

        return CartesianIterationSpace(
            i_interval=oir.Interval(
                start=oir.AxisBound.from_start(min(0, offset.i)),
                end=oir.AxisBound.from_end(max(0, offset.i)),
            ),
            j_interval=oir.Interval(
                start=oir.AxisBound.from_start(min(0, offset.j)),
                end=oir.AxisBound.from_end(max(0, offset.j)),
            ),
        )

    def compose(self, other: "CartesianIterationSpace") -> "CartesianIterationSpace":
        i_interval = oir.Interval(
            start=oir.AxisBound.from_start(
                self.i_interval.start.offset + other.i_interval.start.offset,
            ),
            end=oir.AxisBound.from_end(
                self.i_interval.end.offset + other.i_interval.end.offset,
            ),
        )
        j_interval = oir.Interval(
            start=oir.AxisBound.from_start(
                self.j_interval.start.offset + other.j_interval.start.offset,
            ),
            end=oir.AxisBound.from_end(
                self.j_interval.end.offset + other.j_interval.end.offset,
            ),
        )
        return CartesianIterationSpace(i_interval=i_interval, j_interval=j_interval)

    def __or__(self, other: "CartesianIterationSpace") -> "CartesianIterationSpace":
        i_interval = oir.Interval(
            start=oir.AxisBound.from_start(
                min(
                    self.i_interval.start.offset,
                    other.i_interval.start.offset,
                )
            ),
            end=oir.AxisBound.from_end(
                max(
                    self.i_interval.end.offset,
                    other.i_interval.end.offset,
                )
            ),
        )
        j_interval = oir.Interval(
            start=oir.AxisBound.from_start(
                min(
                    self.j_interval.start.offset,
                    other.j_interval.start.offset,
                )
            ),
            end=oir.AxisBound.from_end(
                max(
                    self.j_interval.end.offset,
                    other.j_interval.end.offset,
                )
            ),
        )
        return CartesianIterationSpace(i_interval=i_interval, j_interval=j_interval)


class CartesianIJIndexSpace(tuple):
    def __new__(cls, *args, **kwargs):
        return super(CartesianIJIndexSpace, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        msg = "CartesianIJIndexSpace must be a pair of pairs of integers."
        if not len(self) == 2:
            raise ValueError(msg)
        if not all(len(v) == 2 for v in self):
            raise ValueError(msg)
        if not all(isinstance(v[0], int) and isinstance(v[1], int) for v in self):
            raise ValueError(msg)

    @staticmethod
    def from_offset(offset: Union[Tuple[int, ...], CartesianOffset]) -> "CartesianIJIndexSpace":
        if isinstance(offset, CartesianOffset):
            return CartesianIJIndexSpace((offset.i, offset.i), (offset.j, offset.j))
        return CartesianIJIndexSpace(((offset[0], offset[0]), (offset[1], offset[1])))

    @staticmethod
    def from_iteration_space(iteration_space: CartesianIterationSpace) -> "CartesianIJIndexSpace":
        return CartesianIJIndexSpace(
            (
                (iteration_space.i_interval.start.offset, iteration_space.i_interval.end.offset),
                (iteration_space.j_interval.start.offset, iteration_space.j_interval.end.offset),
            )
        )

    def compose(
        self, other: Union["CartesianIterationSpace", "CartesianIJIndexSpace"]
    ) -> "CartesianIJIndexSpace":
        if isinstance(other, CartesianIterationSpace):
            other = CartesianIJIndexSpace.from_iteration_space(other)
        return CartesianIJIndexSpace(
            (
                (self[0][0] + other[0][0], (self[0][1] + other[0][1])),
                (self[1][0] + other[1][0], (self[1][1] + other[1][1])),
            )
        )

    def __or__(
        self, other: Union["CartesianIterationSpace", "CartesianIJIndexSpace"]
    ) -> "CartesianIJIndexSpace":
        if isinstance(other, CartesianIterationSpace):
            other = CartesianIJIndexSpace.from_iteration_space(other)
        return CartesianIJIndexSpace(
            (
                (min(self[0][0], other[0][0]), max(self[0][1], other[0][1])),
                (min(self[1][0], other[1][0]), max(self[1][1], other[1][1])),
            )
        )


def oir_iteration_space_computation(stencil: oir.Stencil) -> Dict[int, CartesianIterationSpace]:
    iteration_spaces = dict()

    offsets: Dict[str, List[CartesianOffset]] = dict()
    outputs = set()
    access_spaces: Dict[str, CartesianIterationSpace] = dict()
    # reversed pre_order traversal is post-order but the last nodes per node come first.
    # this is not possible with visitors unless a (reverseed) visit is implemented for every node
    for node in reversed(list(iter_tree(stencil, traversal_order=TraversalOrder.PRE_ORDER))):
        if isinstance(node, oir.FieldAccess):
            if node.name not in offsets:
                offsets[node.name] = list()
            offsets[node.name].append(node.offset)
        elif isinstance(node, oir.AssignStmt):
            if node.left.kind == ExprKind.FIELD:
                outputs.add(node.left.name)
        elif isinstance(node, oir.HorizontalExecution):
            iteration_spaces[id(node)] = CartesianIterationSpace.domain()
            for name in outputs:
                access_space = access_spaces.get(name, CartesianIterationSpace.domain())
                iteration_spaces[id(node)] = iteration_spaces[id(node)] | access_space
            for name in offsets:
                access_space = CartesianIterationSpace.domain()
                for offset in offsets[name]:
                    access_space = access_space | CartesianIterationSpace.from_offset(offset)
                accumulated_extent = iteration_spaces[id(node)].compose(access_space)
                access_spaces[name] = (
                    access_spaces.get(name, CartesianIterationSpace.domain()) | accumulated_extent
                )

            offsets = dict()
            outputs = set()

    return iteration_spaces


def oir_field_boundary_computation(stencil: oir.Stencil) -> Dict[str, CartesianIterationSpace]:
    offsets: Dict[str, List[CartesianOffset]] = dict()
    access_spaces: Dict[str, CartesianIterationSpace] = dict()
    iteration_spaces = oir_iteration_space_computation(stencil)
    for node in iter_tree(stencil, traversal_order=TraversalOrder.POST_ORDER):
        if isinstance(node, oir.FieldAccess):
            if node.name not in offsets:
                offsets[node.name] = list()
            offsets[node.name].append(node.offset)
        elif isinstance(node, oir.HorizontalExecution):
            if iteration_spaces.get(id(node), None) is not None:
                for name in offsets:
                    access_space = CartesianIterationSpace.domain()
                    for offset in offsets[name]:
                        access_space = access_space | CartesianIterationSpace.from_offset(offset)
                    access_spaces[name] = access_spaces.get(
                        name, CartesianIterationSpace.domain()
                    ) | (iteration_spaces[id(node)].compose(access_space))

            offsets = dict()

    return access_spaces


def compute_horizontal_block_extents(node: oir.Stencil) -> Dict[int, Extent]:
    iteration_spaces = oir_iteration_space_computation(node)
    res = {}
    for he_id, it in iteration_spaces.items():
        assert it.i_interval.start.level == common.LevelMarker.START
        assert it.i_interval.end.level == common.LevelMarker.END
        assert it.j_interval.start.level == common.LevelMarker.START
        assert it.j_interval.end.level == common.LevelMarker.END
        res[he_id] = Extent(
            (it.i_interval.start.offset, it.i_interval.end.offset),
            (it.j_interval.start.offset, it.j_interval.end.offset),
            (0, 0),
        )
    return res


class AccessInfoCollector(NodeVisitor):
    def __init__(self, collect_read: bool, collect_write: bool, include_full_domain: bool = False):
        self.collect_read: bool = collect_read
        self.collect_write: bool = collect_write
        self.include_full_domain: bool = include_full_domain

    @dataclass
    class Context:
        axes: Dict[str, List["dcir.Axis"]]
        access_infos: Dict[str, "dcir.FieldAccessInfo"] = field(default_factory=dict)

    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, *, block_extents, ctx=None, **kwargs: Any
    ) -> Dict[str, "dcir.FieldAccessInfo"]:
        if ctx is None:
            ctx = self.Context()
        for section in reversed(node.sections):
            self.visit(section, block_extents=block_extents, ctx=ctx, **kwargs)
        return ctx.access_infos

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        *,
        block_extents,
        ctx=None,
        grid_subset=None,
        **kwargs: Any,
    ) -> Dict[str, "dcir.FieldAccessInfo"]:
        from gtc import daceir as dcir

        if ctx is None:
            ctx = self.Context()

        inner_ctx = self.Context(
            axes=ctx.axes,
        )
        self.visit(node.horizontal_executions, block_extents=block_extents, ctx=inner_ctx, **kwargs)
        inner_infos = inner_ctx.access_infos

        if grid_subset is None or dcir.Axis.K not in grid_subset.intervals:
            k_grid = dcir.GridSubset.from_interval(node.interval, dcir.Axis.K)
        else:
            k_grid = dcir.GridSubset.from_interval(grid_subset.intervals[dcir.Axis.K], dcir.Axis.K)
        inner_infos = {name: info.apply_iteration(k_grid) for name, info in inner_infos.items()}

        ctx.access_infos.update(
            {
                name: info.union(ctx.access_infos.get(name, info))
                for name, info in inner_infos.items()
            }
        )

        return ctx.access_infos

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        block_extents,
        ctx: Context,
        grid_subset=None,
        **kwargs,
    ) -> Dict[str, "dcir.FieldAccessInfo"]:
        from gtc import daceir as dcir

        horizontal_extent = block_extents[id(node)]

        inner_ctx = self.Context(
            axes=ctx.axes,
        )
        self.visit(node.body, horizontal_extent=horizontal_extent, ctx=inner_ctx, **kwargs)
        inner_infos = inner_ctx.access_infos
        ij_grid = dcir.GridSubset.from_gt4py_extent(horizontal_extent)
        if grid_subset is not None:
            for axis in ij_grid.axes():
                if axis in grid_subset.intervals:
                    ij_grid = ij_grid.set_interval(axis, grid_subset.intervals[axis])
        inner_infos = {name: info.apply_iteration(ij_grid) for name, info in inner_infos.items()}

        ctx.access_infos.update(
            {
                name: info.union(ctx.access_infos.get(name, info))
                for name, info in inner_infos.items()
            }
        )

        return ctx.access_infos

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs):
        self.visit(node.right, is_write=False, **kwargs)
        self.visit(node.left, is_write=True, **kwargs)

    def visit_MaskStmt(self, node: oir.MaskStmt, *, is_conditional=False, **kwargs):
        self.generic_visit(node.mask, is_conditional=is_conditional, **kwargs)
        self.generic_visit(node.body, is_conditional=True, **kwargs)

    def _make_access_info(
        self, offset_node: Union[CartesianOffset, oir.VariableKOffset], axes, is_conditional
    ):
        from gtc import daceir as dcir

        offset = list(offset_node.to_tuple())
        if isinstance(offset_node, oir.VariableKOffset):
            variable_offset_axes = [dcir.Axis.K]
        else:
            variable_offset_axes = []

        intervals = dict()
        for axis in axes:
            if axis in variable_offset_axes:
                intervals[axis] = dcir.IndexWithExtent(
                    value=axis.iteration_symbol(), extent=[0, 0], variable=True
                )
            else:
                intervals[axis] = dcir.IndexWithExtent(
                    value=axis.iteration_symbol(),
                    extent=[offset[axis.to_idx()], offset[axis.to_idx()]],
                )
        grid_subset = dcir.GridSubset(intervals=intervals)
        return dcir.FieldAccessInfo(
            grid_subset=grid_subset,
            dynamic_access=len(variable_offset_axes) > 0 or is_conditional,
            variable_offset_axes=variable_offset_axes,
        )

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        is_write: bool = False,
        ctx: "AccessInfoCollector.Context",
        is_conditional=False,
        **kwargs,
    ):
        self.visit(node.offset, is_conditional=is_conditional, ctx=ctx, is_write=False, **kwargs)

        if (not self.collect_read and (not is_write)) or (not self.collect_write and is_write):
            return

        access_info = self._make_access_info(
            node.offset, axes=ctx.axes[node.name], is_conditional=is_conditional
        )
        ctx.access_infos[node.name] = access_info.union(
            ctx.access_infos.get(node.name, access_info)
        )


def compute_dcir_access_infos(
    oir_node,
    *,
    oir_decls=None,
    block_extents=None,
    collect_read=True,
    collect_write=True,
    include_full_domain=False,
    **kwargs,
) -> Dict[str, "dcir.FieldAccessInfo"]:
    from gtc import daceir as dcir

    if block_extents is None:
        assert isinstance(oir_node, oir.Stencil)
        block_extents = compute_horizontal_block_extents(oir_node)

    axes = {
        name: axes_list_from_flags(decl.dimensions)
        for name, decl in oir_decls.items()
        if isinstance(decl, oir.FieldDecl)
    }
    ctx = AccessInfoCollector.Context(axes=axes, access_infos=dict())
    AccessInfoCollector(collect_read=collect_read, collect_write=collect_write).visit(
        oir_node, block_extents=block_extents, ctx=ctx, **kwargs
    )
    if include_full_domain:
        res = dict()
        for name, access_info in ctx.access_infos.items():
            res[name] = access_info.union(
                dcir.FieldAccessInfo(
                    grid_subset=dcir.GridSubset.full_domain(axes=access_info.axes())
                )
            )
    else:
        res = ctx.access_infos

    return res


def make_subset_str(
    context_info: "dcir.FieldAccessInfo", access_info: "dcir.FieldAccessInfo", data_dims
):
    res_strs = []
    clamped_access_info = access_info
    clamped_context_info = context_info
    for axis in access_info.axes():
        if axis in access_info.variable_offset_axes:
            clamped_access_info = clamped_access_info.clamp_full_axis(axis)
            clamped_context_info = clamped_context_info.clamp_full_axis(axis)

    for axis in clamped_access_info.axes():
        context_strs = clamped_context_info.grid_subset.intervals[axis].idx_range
        subset_strs = clamped_access_info.grid_subset.intervals[axis].idx_range
        res_strs.append(
            f"({subset_strs[0]})-({context_strs[0]}):({subset_strs[1]})-({context_strs[0]})"
        )
    res_strs.extend(f"0:{dim}" for dim in data_dims)
    return ",".join(res_strs)


class DaceStrMaker:
    def __init__(self, stencil: oir.Stencil):
        self.decls = {
            decl.name: decl
            for decl in stencil.params + stencil.declarations
            if isinstance(decl, oir.FieldDecl)
        }
        self.block_extents = compute_horizontal_block_extents(stencil)
        self.access_infos = compute_dcir_access_infos(
            stencil,
            oir_decls=self.decls,
            block_extents=self.block_extents,
            collect_read=True,
            collect_write=True,
            include_full_domain=True,
        )
        self.access_collection = AccessCollector.apply(stencil)

    def make_shape(self, field):
        return self.access_infos[field].shape + self.decls[field].data_dims

    def make_input_subset_str(self, node, field):
        local_access_info = compute_dcir_access_infos(
            node,
            collect_read=True,
            collect_write=False,
            block_extents=self.block_extents,
            oir_decls=self.decls,
        )[field]
        for axis in local_access_info.variable_offset_axes:
            local_access_info = local_access_info.clamp_full_axis(axis)

        return self._make_subset_str(local_access_info, field)

    def make_output_subset_str(self, node, field):
        local_access_info = compute_dcir_access_infos(
            node,
            collect_read=False,
            collect_write=True,
            block_extents=self.block_extents,
            oir_decls=self.decls,
        )[field]
        for axis in local_access_info.variable_offset_axes:
            local_access_info = local_access_info.clamp_full_axis(axis)

        return self._make_subset_str(local_access_info, field)

    def _make_subset_str(self, local_access_info, field):
        global_access_info = self.access_infos[field]
        return make_subset_str(global_access_info, local_access_info, self.decls[field].data_dims)


def union_node_access_infos(nodes: List[eve.Node]):
    from gtc import daceir as dcir

    read_accesses: Dict[str, dcir.FieldAccessInfo] = dict()
    write_accesses: Dict[str, dcir.FieldAccessInfo] = dict()
    for maybe_statenode in nodes:
        for node in maybe_statenode.iter_tree().if_isinstance(dcir.ComputationNode):
            read_accesses.update(
                {
                    name: access_info.union(read_accesses.get(name, access_info))
                    for name, access_info in node.read_accesses.items()
                }
            )
            write_accesses.update(
                {
                    name: access_info.union(write_accesses.get(name, access_info))
                    for name, access_info in node.write_accesses.items()
                }
            )

    return read_accesses, write_accesses, union_access_info_dicts(read_accesses, write_accesses)


def union_access_info_dicts(
    first_infos: Dict[str, "dcir.FieldAccessInfo"], second_infos: Dict[str, "dcir.FieldAccessInfo"]
):
    res = dict(first_infos)
    for key, access_info in second_infos.items():
        res[key] = access_info.union(first_infos.get(key, access_info))
    return res


def flatten_list(list_or_node: Union[List[Any], eve.Node]) -> List[eve.Node]:
    list_or_node = [list_or_node]
    while not all(isinstance(ref, eve.Node) for ref in list_or_node):
        list_or_node = [r for li in list_or_node for r in li]

    return list_or_node
