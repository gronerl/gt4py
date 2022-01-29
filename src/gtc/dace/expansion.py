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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import dace
import dace.library
import dace.subsets

import eve
import gtc.common as common
import gtc.daceir as dir
import gtc.oir as oir
from eve import NodeTranslator, NodeVisitor, codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from gt4py.definitions import Extent
from gtc import daceir as dir
from gtc.common import LoopOrder
from gtc.dace.nodes import StencilComputation
from gtc.dace.utils import (
    array_dimensions,
    get_access_collection,
    get_axis_bound_diff_str,
    get_axis_bound_str,
    get_interval_range_str,
    get_tasklet_symbol,
)
from gtc.oir import Interval
from gtc.passes.oir_optimizations.utils import AccessCollector

from .utils import compute_stencil_field_extents, compute_stencil_field_intervals


class TaskletCodegen(codegen.TemplatedGenerator):

    ScalarAccess = as_fmt("{name}")

    def visit_FieldAccess(self, node: oir.FieldAccess, *, is_target, targets):

        if (is_target or node.name in targets) and self.visit(node.offset) == "":
            targets.add(node.name)
            name = "__" + node.name
        else:
            name = node.name + "__" + self.visit(node.offset)
        if node.data_index:
            offset_str = "[" + ",".join(self.visit(node.data_index)) + "]"
        else:
            offset_str = ""
        return name + offset_str

    def visit_CartesianOffset(self, node: common.CartesianOffset):
        res = []
        if node.i != 0:
            res.append(f'i{"m" if node.i<0 else "p"}{abs(node.i):d}')
        if node.j != 0:
            res.append(f'j{"m" if node.j<0 else "p"}{abs(node.j):d}')
        if node.k != 0:
            res.append(f'k{"m" if node.k<0 else "p"}{abs(node.k):d}')
        return "_".join(res)

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs):
        right = self.visit(node.right, is_target=False, **kwargs)
        left = self.visit(node.left, is_target=True, **kwargs)
        return f"{left} = {right}"

    BinaryOp = as_fmt("({left} {op} {right})")

    UnaryOp = as_fmt("({op}{expr})")

    TernaryOp = as_fmt("({true_expr} if {cond} else {false_expr})")

    def visit_BuiltInLiteral(self, builtin: common.BuiltInLiteral, **kwargs: Any) -> str:
        if builtin == common.BuiltInLiteral.TRUE:
            return "True"
        elif builtin == common.BuiltInLiteral.FALSE:
            return "False"
        raise NotImplementedError("Not implemented BuiltInLiteral encountered.")

    Literal = as_fmt("{value}")

    Cast = as_fmt("{dtype}({expr})")

    def visit_NativeFunction(self, func: common.NativeFunction, **kwargs: Any) -> str:
        try:
            return {
                common.NativeFunction.ABS: "abs",
                common.NativeFunction.MIN: "min",
                common.NativeFunction.MAX: "max",
                common.NativeFunction.MOD: "fmod",
                common.NativeFunction.SIN: "dace.math.sin",
                common.NativeFunction.COS: "dace.math.cos",
                common.NativeFunction.TAN: "dace.math.tan",
                common.NativeFunction.ARCSIN: "asin",
                common.NativeFunction.ARCCOS: "acos",
                common.NativeFunction.ARCTAN: "atan",
                common.NativeFunction.SINH: "dace.math.sinh",
                common.NativeFunction.COSH: "dace.math.cosh",
                common.NativeFunction.TANH: "dace.math.tanh",
                common.NativeFunction.ARCSINH: "asinh",
                common.NativeFunction.ARCCOSH: "acosh",
                common.NativeFunction.ARCTANH: "atanh",
                common.NativeFunction.SQRT: "dace.math.sqrt",
                common.NativeFunction.POW: "dace.math.pow",
                common.NativeFunction.EXP: "dace.math.exp",
                common.NativeFunction.LOG: "dace.math.log",
                common.NativeFunction.GAMMA: "tgamma",
                common.NativeFunction.CBRT: "cbrt",
                common.NativeFunction.ISFINITE: "isfinite",
                common.NativeFunction.ISINF: "isinf",
                common.NativeFunction.ISNAN: "isnan",
                common.NativeFunction.FLOOR: "dace.math.ifloor",
                common.NativeFunction.CEIL: "ceil",
                common.NativeFunction.TRUNC: "trunc",
            }[func]
        except KeyError as error:
            raise NotImplementedError("Not implemented NativeFunction encountered.") from error

    NativeFuncCall = as_mako("${func}(${','.join(args)})")

    def visit_DataType(self, dtype: common.DataType, **kwargs: Any) -> str:
        if dtype == common.DataType.BOOL:
            return "dace.bool_"
        elif dtype == common.DataType.INT8:
            return "dace.int8"
        elif dtype == common.DataType.INT16:
            return "dace.int16"
        elif dtype == common.DataType.INT32:
            return "dace.int32"
        elif dtype == common.DataType.INT64:
            return "dace.int64"
        elif dtype == common.DataType.FLOAT32:
            return "dace.float32"
        elif dtype == common.DataType.FLOAT64:
            return "dace.float64"
        raise NotImplementedError("Not implemented DataType encountered.")

    def visit_UnaryOperator(self, op: common.UnaryOperator, **kwargs: Any) -> str:
        if op == common.UnaryOperator.NOT:
            return " not "
        elif op == common.UnaryOperator.NEG:
            return "-"
        elif op == common.UnaryOperator.POS:
            return "+"
        raise NotImplementedError("Not implemented UnaryOperator encountered.")

    Arg = as_fmt("{name}")

    Param = as_fmt("{name}")

    LocalScalar = as_fmt("{name}: {dtype}")

    def visit_Tasklet(self, node: dir.Tasklet):
        return "\n".join(self.visit(node.stmts, targets=set()))

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs):
        mask_str = ""
        indent = ""
        if node.mask is not None:
            mask_str = f"if {self.visit(node.mask, is_target=False, **kwargs)}:"
            indent = "    "
        body_code = self.visit(node.body, targets=kwargs["targets"])
        body_code = [indent + b for b in body_code]
        return "\n".join([mask_str] + body_code)

    class RemoveCastInIndexVisitor(eve.NodeTranslator):
        def visit_FieldAccess(self, node: oir.FieldAccess):
            if node.data_index:
                return self.generic_visit(node, in_idx=True)
            else:
                return self.generic_visit(node)

        def visit_Cast(self, node: oir.Cast, in_idx=False):
            if in_idx:
                return node.expr
            else:
                return node

        def visit_Literal(self, node: oir.Cast, in_idx=False):
            if in_idx:
                return node
            else:
                return oir.Cast(dtype=node.dtype, expr=node)

    @classmethod
    def apply(cls, node: oir.HorizontalExecution, **kwargs: Any) -> str:
        preprocessed_node = cls.RemoveCastInIndexVisitor().visit(node)
        if not isinstance(node, oir.HorizontalExecution):
            raise ValueError("apply() requires oir.HorizontalExecution node")
        generated_code = super().apply(preprocessed_node)
        formatted_code = codegen.format_source("python", generated_code)
        return formatted_code


class DaCeIRBuilder(NodeTranslator):
    @dataclass
    class GlobalContext:
        library_node: StencilComputation
        block_extents: Dict[int, Extent]

    @dataclass
    class SDFGContext:
        field_extents: Dict[str, dir.FieldExtent]

        @classmethod
        def from_gt4py_field_extents_and_intervals(cls, field_extents, field_intervals):
            return cls(
                field_extents={
                    field: dir.FieldExtent.from_gt4py_extent_and_interval(
                        extent, field_intervals[field]
                    )
                    for field, extent in field_extents.items()
                }
            )

    @dataclass
    class IterationContext:
        grid_subset: dir.GridSubset

        def restrict_grid_subset(self, axis: dir.Axis):
            grid_subset = self.grid_subset.restrict(axis)
            return DaCeIRBuilder.IterationContext(
                grid_subset=grid_subset,
            )

    def visit_HorizontalExecution(
        self, node: oir.HorizontalExecution, *, global_ctx, iteration_ctx, **kwargs
    ):
        extent = global_ctx.block_extents[id(node)]
        decls = [self.visit(decl) for decl in node.declarations]
        stmts = [self.visit(stmt) for stmt in node.body]

        out_fields = AccessCollector.apply(node).write_fields()

        jmap_grid_subset = iteration_ctx.grid_subset.restrict(
            dir.Axis.J,
        )
        imap_grid_subset = iteration_ctx.grid_subset.restrict(
            dir.Axis.I,
        )
        irange = dir.Range.from_axis_and_extent(dir.Axis.I, extent)
        i_read_extents = dir.make_read_extents(
            node, grid_subset=imap_grid_subset, block_extents=global_ctx.block_extents
        )

        imap = dir.DomainMap(
            index_range=irange,
            computations=[
                dir.Tasklet(stmts=decls + stmts, read_extents=i_read_extents, out_fields=out_fields)
            ],
            grid_subset=imap_grid_subset,
            read_extents=i_read_extents,
            out_fields=out_fields,
        )
        j_read_extents = dir.make_read_extents(
            node, grid_subset=jmap_grid_subset, block_extents=global_ctx.block_extents
        )
        jrange = dir.Range.from_axis_and_extent(dir.Axis.J, extent)
        return dir.DomainMap(
            index_range=jrange,
            computations=[imap],
            grid_subset=jmap_grid_subset,
            read_extents=j_read_extents,
            out_fields=out_fields,
        )

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        *,
        loop_order,
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        global_ctx: "DaCeIRBuilder.GlobalContext",
        **kwargs,
    ):
        iteration_ctx = iteration_ctx.restrict_grid_subset(dir.Axis.K)
        scope_nodes = [
            self.visit(he, iteration_ctx=iteration_ctx, global_ctx=global_ctx, **kwargs)
            for he in node.horizontal_executions
        ]
        access_collection = AccessCollector.apply(node)
        grid_subset = iteration_ctx.grid_subset.restrict(dir.Axis.K)
        if False and loop_order == common.LoopOrder.PARALLEL:

            read_extents = dir.make_read_extents(
                node,
                grid_subset=grid_subset,
                block_extents=global_ctx.block_extents,
            )
            return dir.DomainMap(
                index_range=dir.Range(
                    var=dir.Axis.K.iteration_symbol(),
                    start=dir.AxisBound.from_common(dir.Axis.K, node.interval.start),
                    end=dir.AxisBound.from_common(dir.Axis.K, node.interval.end),
                    stride=1,
                ),
                computations=scope_nodes,
                grid_subset=grid_subset,
                read_extents=read_extents,
                out_fields=access_collection.write_fields(),
            )
        else:
            stride = 1 if loop_order == common.LoopOrder.FORWARD else -1
            return dir.DomainLoop(
                index_range=dir.Range(
                    var=dir.Axis.K.iteration_symbol(),
                    start=dir.AxisBound.from_common(dir.Axis.K, node.interval.start),
                    end=dir.AxisBound.from_common(dir.Axis.K, node.interval.end),
                    stride=stride,
                ),
                loop_states=[
                    dir.ComputationState(computations=[scope_node], grid_subset=grid_subset)
                    for scope_node in scope_nodes
                ],
                grid_subset=grid_subset,
            )

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        *,
        global_ctx: "DaCeIRBuilder.GlobalContext",
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        **kwargs,
    ):
        sdfg_ctx = DaCeIRBuilder.SDFGContext.from_gt4py_field_extents_and_intervals(
            compute_stencil_field_extents(node, block_extents=global_ctx.block_extents),
            compute_stencil_field_intervals(node),
        )

        computations = [
            self.visit(
                section,
                loop_order=node.loop_order,
                sdfg_ctx=sdfg_ctx,
                global_ctx=global_ctx,
                iteration_ctx=iteration_ctx,
                **kwargs,
            )
            for section in node.sections
        ]
        read_extents = dir.make_read_extents(
            node, grid_subset=iteration_ctx.grid_subset, block_extents=global_ctx.block_extents
        )
        if False and node.loop_order == common.LoopOrder.PARALLEL:
            return dir.StateMachine(
                states=[
                    dir.ComputationState(
                        computations=computations, grid_subset=iteration_ctx.grid_subset
                    )
                ],
                field_extents=sdfg_ctx.field_extents,
                read_extents=read_extents,
                out_fields=AccessCollector.apply(node).write_fields(),
            )
        else:
            return dir.StateMachine(
                states=computations,
                field_extents=sdfg_ctx.field_extents,
                read_extents=read_extents,
                out_fields=AccessCollector.apply(node).write_fields(),
            )


class StencilComputationSDFGBuilder(NodeVisitor):
    @dataclass
    class NodeContext:
        input_node_and_conns: Dict[str, Tuple[dace.nodes.Node, Optional[str]]]
        output_node_and_conns: Dict[str, Tuple[dace.nodes.Node, Optional[str]]]

    @dataclass
    class SDFGContext:
        sdfg: dace.SDFG
        state: dace.SDFGState
        field_extents: Optional[Dict[str, dir.FieldExtent]] = None

        def add_state(self):
            sink = self.sdfg.sink_nodes()[0]
            state = self.sdfg.add_state()
            self.sdfg.add_edge(sink, state, dace.InterstateEdge())

            return StencilComputationSDFGBuilder.SDFGContext(
                sdfg=self.sdfg, state=state, field_extents=self.field_extents
            )

        def add_loop(self, range: dir.Range):
            sink = self.sdfg.sink_nodes()[0]
            loop_state = self.sdfg.add_state()
            self.sdfg.add_loop(
                before_state=sink,
                loop_state=loop_state,
                after_state=self.sdfg.add_state(),
                loop_var=range.var,
                initialize_expr=str(range.start),
                condition_expr=f"{range.var} != {range.end}",
                increment_expr=f"{range.stride}",
            )
            return StencilComputationSDFGBuilder.SDFGContext(
                sdfg=self.sdfg, state=loop_state, field_extents=self.field_extents
            )

    @staticmethod
    def _interval_or_idx_range(node: Union[dir.Interval, dir.IndexWithExtent]) -> Tuple[str, str]:
        if isinstance(node, dir.Interval):
            return str(node.start), str(node.end)
        else:
            return f"{node.value}{node.extent[0]:+d}", f"{node.value}{node.extent[1]:+d}"

    @staticmethod
    def _make_subset_str(context_extent: dir.FieldExtent, subset_extent: dir.FieldExtent):
        res_strs = []
        for axis in [dir.Axis.I, dir.Axis.J, dir.Axis.K]:
            context_strs = StencilComputationSDFGBuilder._interval_or_idx_range(
                context_extent.values[axis]
            )
            subset_strs = StencilComputationSDFGBuilder._interval_or_idx_range(
                subset_extent.values[axis]
            )
            res_strs.append(
                f"({subset_strs[1]})-({context_strs[0]}):({subset_strs[1]})-({context_strs[0]})+1"
            )

        return ",".join(res_strs)

    @staticmethod
    def _get_memlets(
        node: dir.ComputationNode,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
    ):

        in_memlets = dict()
        for field, extent in node.read_extents.items():
            context_extent = sdfg_ctx.field_extents[field]
            in_memlets["IN_" + field] = dace.Memlet.simple(
                field,
                subset_str=StencilComputationSDFGBuilder._make_subset_str(context_extent, extent),
            )

        out_memlets = dict()
        for field in node.out_fields:
            extent = node.grid_subset.to_field_extent()
            context_extent = sdfg_ctx.field_extents[field]
            out_memlets["OUT_" + field] = dace.Memlet.simple(
                field,
                subset_str=StencilComputationSDFGBuilder._make_subset_str(context_extent, extent),
            )

        return in_memlets, out_memlets

    @staticmethod
    def _add_edges(
        entry_node: dace.nodes.Node,
        exit_node: dace.nodes.Node,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        in_memlets: Dict[str, dace.Memlet],
        out_memlets: Dict[str, dace.Memlet],
    ):
        for conn, memlet in in_memlets.items():
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[memlet.data], entry_node, conn, memlet
            )
        for conn, memlet in out_memlets.items():
            sdfg_ctx.state.add_edge(
                exit_node, conn, *node_ctx.output_node_and_conns[memlet.data], memlet
            )

    @staticmethod
    def _add_origin(
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        field: str,
        subset: Union[dace.subsets.Range, str],
    ):
        if isinstance(subset, str):
            subset = dace.subsets.Range.from_string(subset)
        origin_strs = []
        field_extent = sdfg_ctx.field_extents[field]
        for axis in dir.Axis.I, dir.Axis.J, dir.Axis.K:
            extent = field_extent[axis]
            if isinstance(extent, dir.Interval):
                origin_strs.append(f"-({get_axis_bound_str(extent.start, axis.domain_symbol())})")
            else:
                assert isinstance(extent, dir.IndexWithExtent)
                origin_strs.append(f"-({extent.value})")

        sym = dace.symbolic.pystr_to_symbolic
        res_ranges = [
            (sym(f"({rng[0]})+({orig})"), sym(f"({rng[1]})+({orig})"), rng[2])
            for rng, orig in zip(subset.ranges, origin_strs)
        ]
        return dace.subsets.Range(res_ranges)

    def visit_Tasklet(
        self,
        node: dir.Tasklet,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
    ):
        code = TaskletCodegen().visit(node)
        access_collection = AccessCollector.apply(node)
        in_memlets = dict()
        for field, offsets in access_collection.read_offsets().items():
            for offset in offsets:
                conn_name = get_tasklet_symbol(field, offset, is_target=False)
                subset_str = "i{:+d},j{:+d},k{:+d}".format(*offset)
                subset = StencilComputationSDFGBuilder._add_origin(
                    sdfg_ctx=sdfg_ctx, field=field, subset=subset_str
                )
                in_memlets[conn_name] = dace.Memlet.simple(field, subset_str=subset)

        out_memlets = dict()
        for field in access_collection.write_fields():
            conn_name = get_tasklet_symbol(field, (0, 0, 0), is_target=True)
            subset = StencilComputationSDFGBuilder._add_origin(
                sdfg_ctx=sdfg_ctx, field=field, subset="i,j,k"
            )
            out_memlets[conn_name] = dace.Memlet.simple(field, subset_str=subset)

        tasklet = dace.nodes.Tasklet(
            label=f"Tasklet_{id(node)}",
            code=code,
            inputs={inp for inp in in_memlets.keys()},
            outputs={outp for outp in out_memlets.keys()},
        )

        sdfg_ctx.state.add_node(tasklet)

        StencilComputationSDFGBuilder._add_edges(
            tasklet,
            tasklet,
            sdfg_ctx=sdfg_ctx,
            in_memlets=in_memlets,
            out_memlets=out_memlets,
            node_ctx=node_ctx,
        )

    def visit_Range(self, node: dir.Range, **kwargs):
        return node.to_ndrange()

    def visit_DomainMap(
        self,
        node: dir.DomainMap,
        *,
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
    ):
        map_entry, map_exit = sdfg_ctx.state.add_map(
            name=node.index_range.var + "_map", ndrange=self.visit(node.index_range)
        )

        input_node_and_conns: Dict[str, Tuple[dace.nodes.Node, str]] = dict()
        output_node_and_conns: Dict[str, Tuple[dace.nodes.Node, str]] = dict()
        for scope_node in node.computations:
            access_collection = AccessCollector.apply(scope_node)
            for field in access_collection.read_fields():
                map_entry.add_in_connector("IN_" + field)
                map_entry.add_out_connector("OUT_" + field)
                input_node_and_conns[field] = (map_entry, "OUT_" + field)
            for field in access_collection.write_fields():
                map_exit.add_in_connector("IN_" + field)
                map_exit.add_out_connector("OUT_" + field)
                output_node_and_conns[field] = (map_exit, "IN_" + field)

            inner_node_ctx = StencilComputationSDFGBuilder.NodeContext(
                input_node_and_conns=input_node_and_conns,
                output_node_and_conns=output_node_and_conns,
            )

            self.visit(scope_node, sdfg_ctx=sdfg_ctx, node_ctx=inner_node_ctx)

        in_memlets, out_memlets = self._get_memlets(node, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)

        self._add_edges(
            map_entry,
            map_exit,
            node_ctx=node_ctx,
            sdfg_ctx=sdfg_ctx,
            in_memlets=in_memlets,
            out_memlets=out_memlets,
        )

    def visit_DomainLoop(
        self, node: dir.DomainLoop, *, sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext"
    ):
        sdfg_ctx = sdfg_ctx.add_loop(node.index_range)
        self.visit(node.loop_states, sdfg_ctx=sdfg_ctx)

    def visit_ComputationState(
        self,
        node: dir.ComputationState,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
    ):
        sdfg_ctx = sdfg_ctx.add_state()

        for computation in node.computations:
            read_acc_and_conn = dict()
            write_acc_and_conn = dict()
            assert isinstance(computation, dir.ComputationNode)
            for field in computation.read_extents.keys():
                if field not in read_acc_and_conn:
                    read_acc_and_conn[field] = (sdfg_ctx.state.add_access(field), None)
            for field in computation.out_fields:
                if field not in write_acc_and_conn:
                    write_acc_and_conn[field] = (sdfg_ctx.state.add_access(field), None)
            node_ctx = StencilComputationSDFGBuilder.NodeContext(
                input_node_and_conns=read_acc_and_conn,
                output_node_and_conns=write_acc_and_conn,
            )
            self.visit(computation, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)

    def visit_StateMachine(
        self,
        node: dir.StateMachine,
        *,
        sdfg_ctx: Optional["StencilComputationSDFGBuilder.SDFGContext"] = None,
        node_ctx: Optional["StencilComputationSDFGBuilder.NodeContext"] = None,
    ):

        sdfg = dace.SDFG(f"StateMachine_{id(node)}")
        state = sdfg.add_state()

        nsdfg = dace.nodes.NestedSDFG(
            label=sdfg.label,
            sdfg=sdfg,
            inputs=set(node.read_extents.keys()),
            outputs=node.out_fields,
        )

        sdfg_ctx = StencilComputationSDFGBuilder.SDFGContext(
            sdfg=sdfg,
            state=state,
            field_extents=node.field_extents,
        )

        for computation_state in node.states:
            self.visit(computation_state, sdfg_ctx=sdfg_ctx)

        if sdfg_ctx is not None and node_ctx is not None:
            in_memlets, out_memlets = StencilComputationSDFGBuilder._get_memlets(
                node, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
            )

            StencilComputationSDFGBuilder._add_edges(
                nsdfg,
                nsdfg,
                sdfg_ctx=sdfg_ctx,
                node_ctx=node_ctx,
                in_memlets=in_memlets,
                out_memlets=out_memlets,
            )
        sdfg.view()


@dace.library.register_expansion(StencilComputation, "default")
class StencilComputationExpansion(dace.library.ExpandTransformation):
    environments: List = []

    @staticmethod
    def expansion(
        node: "StencilComputation", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.nodes.NestedSDFG:

        start, end = (
            node.oir_node.sections[0].interval.start,
            node.oir_node.sections[0].interval.end,
        )
        for section in node.oir_node.sections:
            start = min(start, section.interval.start)
            end = max(end, section.interval.end)

        overall_interval = dir.Interval(
            start=dir.AxisBound(axis=dir.Axis.K, level=start.level, offset=start.offset),
            end=dir.AxisBound(axis=dir.Axis.K, level=end.level, offset=end.offset),
        )
        overall_extent = Extent.zeros(ndims=3)
        for he in node.oir_node.iter_tree().if_isinstance(oir.HorizontalExecution):
            overall_extent = overall_extent.union(node.extents[id(he)])
        iteration_ctx = DaCeIRBuilder.IterationContext(
            grid_subset=dir.GridSubset.from_gt4py_extent_and_interval(
                overall_extent, overall_interval
            )
        )
        daceir_builder_global_ctx = DaCeIRBuilder.GlobalContext(
            library_node=node, block_extents=node.extents
        )

        daceir = DaCeIRBuilder().visit(
            node.oir_node, global_ctx=daceir_builder_global_ctx, iteration_ctx=iteration_ctx
        )

        #
        nsdfg = StencilComputationSDFGBuilder().visit(daceir)
        # TODO fix context memlets
        # TODO remove connector prefixes
        return nsdfg
