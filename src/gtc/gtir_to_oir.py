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
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from eve import NodeTranslator, NodeVisitor
from gtc import gtir, oir
from gtc.common import CartesianOffset, DataType, LogicalOperator, UnaryOperator
from gtc.passes.oir_optimizations.utils import AccessCollector


def _create_mask(ctx: "GTIRToOIR.Context", name: str, cond: oir.Expr) -> oir.Temporary:
    mask_field_decl = oir.Temporary(name=name, dtype=DataType.BOOL)
    ctx.add_decl(mask_field_decl)

    fill_mask_field = oir.HorizontalExecution(
        body=[
            oir.AssignStmt(
                left=oir.FieldAccess(
                    name=mask_field_decl.name,
                    offset=CartesianOffset.zero(),
                    dtype=mask_field_decl.dtype,
                ),
                right=cond,
            )
        ],
        declarations=[],
    )
    ctx.add_horizontal_execution(fill_mask_field)
    return mask_field_decl


def gtir_to_oir(gtir: gtir.Stencil) -> oir.Stencil:
    oir_no_iteration_space = GTIRToOIR().visit(gtir)
    stencil = OIRIterationSpaceTranslator.apply(oir_no_iteration_space)
    return stencil


class GTIRToOIR(NodeTranslator):
    @dataclass
    class Context:
        """
        Context for Stmts.

        `Stmt` nodes create `Temporary` nodes and `HorizontalExecution` nodes.
        All visit()-methods for `Stmt` have no return value,
        they attach their result to the Context object.
        """

        decls: List = field(default_factory=list)
        horizontal_executions: List = field(default_factory=list)

        def add_decl(self, decl: oir.Decl) -> "GTIRToOIR.Context":
            self.decls.append(decl)
            return self

        def add_horizontal_execution(
            self, horizontal_execution: oir.HorizontalExecution
        ) -> "GTIRToOIR.Context":
            self.horizontal_executions.append(horizontal_execution)
            return self

    def visit_ParAssignStmt(
        self, node: gtir.ParAssignStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs: Any
    ) -> None:
        ctx.add_horizontal_execution(
            oir.HorizontalExecution(
                body=[oir.AssignStmt(left=self.visit(node.left), right=self.visit(node.right))],
                mask=mask,
                declarations=[],
            ),
        )

    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs: Any) -> oir.FieldAccess:
        return oir.FieldAccess(name=node.name, offset=node.offset, dtype=node.dtype)

    def visit_ScalarAccess(self, node: gtir.ScalarAccess, **kwargs: Any) -> oir.ScalarAccess:
        return oir.ScalarAccess(name=node.name, dtype=node.dtype)

    def visit_Literal(self, node: gtir.Literal, **kwargs: Any) -> oir.Literal:
        return oir.Literal(value=self.visit(node.value), dtype=node.dtype, kind=node.kind)

    def visit_UnaryOp(self, node: gtir.UnaryOp, **kwargs: Any) -> oir.UnaryOp:
        return oir.UnaryOp(op=node.op, expr=self.visit(node.expr))

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs: Any) -> oir.BinaryOp:
        return oir.BinaryOp(op=node.op, left=self.visit(node.left), right=self.visit(node.right))

    def visit_TernaryOp(self, node: gtir.TernaryOp, **kwargs: Any) -> oir.TernaryOp:
        return oir.TernaryOp(
            cond=self.visit(node.cond),
            true_expr=self.visit(node.true_expr),
            false_expr=self.visit(node.false_expr),
        )

    def visit_Cast(self, node: gtir.Cast, **kwargs: Any) -> oir.Cast:
        return oir.Cast(dtype=node.dtype, expr=self.visit(node.expr, **kwargs))

    def visit_FieldDecl(self, node: gtir.FieldDecl, **kwargs: Any) -> oir.FieldDecl:
        return oir.FieldDecl(name=node.name, dtype=node.dtype)

    def visit_ScalarDecl(self, node: gtir.ScalarDecl, **kwargs: Any) -> oir.ScalarDecl:
        return oir.ScalarDecl(name=node.name, dtype=node.dtype)

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall, **kwargs: Any) -> oir.NativeFuncCall:
        return oir.NativeFuncCall(
            func=node.func, args=self.visit(node.args), dtype=node.dtype, kind=node.kind
        )

    def visit_FieldIfStmt(
        self, node: gtir.FieldIfStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs: Any
    ) -> None:
        mask_field_decl = _create_mask(ctx, f"mask_{node.id_}", self.visit(node.cond))
        current_mask = oir.FieldAccess(
            name=mask_field_decl.name, offset=CartesianOffset.zero(), dtype=mask_field_decl.dtype
        )
        combined_mask = current_mask
        if mask:
            combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=combined_mask)
        self.visit(node.true_branch.body, mask=combined_mask, ctx=ctx)

        if node.false_branch:
            combined_mask = oir.UnaryOp(op=UnaryOperator.NOT, expr=current_mask)
            if mask:
                combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=combined_mask)
            self.visit(
                node.false_branch.body,
                mask=combined_mask,
                ctx=ctx,
            )

    # For now we represent ScalarIf (and FieldIf) both as masks on the HorizontalExecution.
    # This is not meant to be set in stone...
    def visit_ScalarIfStmt(
        self, node: gtir.ScalarIfStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs: Any
    ) -> None:
        current_mask = self.visit(node.cond)
        combined_mask = current_mask
        if mask:
            combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=current_mask)

        self.visit(node.true_branch.body, mask=combined_mask, ctx=ctx)
        if node.false_branch:
            combined_mask = oir.UnaryOp(op=UnaryOperator.NOT, expr=current_mask)
            if mask:
                combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=combined_mask)
            self.visit(
                node.false_branch.body,
                mask=combined_mask,
                ctx=ctx,
            )

    def visit_Interval(self, node: gtir.Interval, **kwargs: Any) -> oir.Interval:
        return oir.Interval(
            start=self.visit(node.start),
            end=self.visit(node.end),
        )

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        ctx = self.Context()
        self.visit(node.body, ctx=ctx)

        # should temporaries live at this level?
        for temp in node.temporaries:
            ctx.add_decl(oir.Temporary(name=temp.name, dtype=temp.dtype))

        return oir.VerticalLoop(
            interval=self.visit(node.interval),
            loop_order=node.loop_order,
            declarations=ctx.decls,
            horizontal_executions=ctx.horizontal_executions,
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> oir.Stencil:
        vertical_loops = self.visit(node.vertical_loops)

        return oir.Stencil(
            name=node.name,
            params=self.visit(node.params),
            vertical_loops=vertical_loops,
        )


class OIRHorizontalExecutionDependencyGraphBuilder(NodeVisitor):
    class ReadWriteContext:
        writes: Dict[str, oir.IntervalMapping]
        accesses: Dict[str, oir.IntervalMapping]

        def __init__(self) -> None:
            self.writes = dict()
            self.accesses = dict()

        def set_write(
            self, name: str, interval: oir.Interval, node: oir.HorizontalExecution
        ) -> None:
            if name not in self.writes:
                self.writes[name] = oir.IntervalMapping()
            if name not in self.accesses:
                self.accesses[name] = oir.IntervalMapping()
            self.writes[name][interval] = node.id_
            self.accesses[name][interval] = node.id_

        def set_read(
            self, name: str, interval: oir.Interval, node: oir.HorizontalExecution
        ) -> None:
            if name not in self.accesses:
                self.accesses[name] = oir.IntervalMapping()
            self.accesses[name][interval] = node.id_

        def get_writes(self, name: str, interval: oir.Interval) -> Set[Any]:
            if name in self.writes:
                return set(self.writes[name][interval])
            else:
                return set()

        def get_accesses(self, name: str, interval: oir.Interval) -> Set[Any]:
            if name in self.accesses:
                return set(self.accesses[name][interval])
            else:
                return set()

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        interval: oir.Interval,
        graph: nx.Graph,
        context: "OIRHorizontalExecutionDependencyGraphBuilder.ReadWriteContext",
    ) -> None:
        graph.add_node(node.id_, node=node, interval=interval)

        access_collection = AccessCollector.apply(node)

        read_k_intervals: Dict[str, oir.Interval] = dict()
        for name, offsets in access_collection.read_offsets().items():
            offsets_iter = iter(offsets)
            o = next(offsets_iter)
            min_k_offset = o[2]
            max_k_offset = o[2]
            for o in offsets_iter:
                min_k_offset = min(o[2], min_k_offset) if min_k_offset is not None else o[2]
                max_k_offset = max(o[2], max_k_offset) if max_k_offset is not None else o[2]
            start = oir.AxisBound(
                level=interval.start.level, offset=interval.start.offset + min_k_offset
            )
            end = oir.AxisBound(level=interval.end.level, offset=interval.end.offset + max_k_offset)
            read_k_intervals[name] = oir.Interval(start=start, end=end)

        dependencies: Dict[str, Set[oir.HorizontalExecution]] = dict()
        for name, read_interval in read_k_intervals.items():
            for n in context.get_writes(name, read_interval):
                if name not in dependencies:
                    dependencies[name] = set()
                dependencies[name].update([n])
        for name in access_collection.write_fields():
            for n in context.get_accesses(name, interval):
                if name not in dependencies:
                    dependencies[name] = set()
                dependencies[name].update([n])

        for ids in dependencies.values():
            for id_ in ids:
                graph.add_edge(id_, node.id_)

        for name, read_interval in read_k_intervals.items():
            context.set_read(name, read_interval, node)
        for name in access_collection.write_fields():
            context.set_write(name, interval, node)

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Dict[str, Any]) -> None:
        interval = node.interval
        self.generic_visit(node, interval=interval, **kwargs)

    def visit_Stencil(self, node: oir.Stencil, *, graph: nx.DiGraph) -> None:
        context = OIRHorizontalExecutionDependencyGraphBuilder.ReadWriteContext()
        self.generic_visit(node, graph=graph, context=context)

    @classmethod
    def apply(cls, node: oir.Stencil) -> nx.DiGraph:
        visitor = cls()
        graph = nx.DiGraph()
        visitor.visit(node, graph=graph)
        return graph


def _dependency_expansion_backward(
    read_node: oir.HorizontalExecution,
    read_node_interval: oir.Interval,
    write_node: oir.HorizontalExecution,
    write_node_interval: oir.Interval,
    iteration_offsets: Dict[str, Optional[oir.CartesianIterationOffset]],
) -> None:
    read_node_access_collection = AccessCollector.apply(read_node)
    write_node_access_collection = AccessCollector.apply(write_node)

    read_iteration_offset = iteration_offsets.get(str(read_node.id_), None)
    if read_iteration_offset is not None:

        for name in (
            read_node_access_collection.read_fields() & write_node_access_collection.write_fields()
        ):
            read_offsets = read_node_access_collection.read_offsets()[name]

            is_write_before_read = False
            collections = []
            if read_node.mask is not None:
                mask_access_collection = AccessCollector.Result([])
                AccessCollector().visit(
                    read_node.mask,
                    accesses=mask_access_collection._ordered_accesses,
                    is_write=False,
                )
                collections.append(mask_access_collection)

            for stmt in read_node.body:
                stmt_access_collection = AccessCollector.Result([])
                AccessCollector().visit(stmt, accesses=stmt_access_collection._ordered_accesses)
                collections.append(stmt_access_collection)

            for collection in collections:
                if name in collection.read_fields():
                    break
                elif name in collection.write_fields():
                    is_write_before_read = True
            if is_write_before_read and all(ro[2] == 0 for ro in read_offsets):
                continue

            for offset in read_offsets:
                read_interval = read_node_interval.shift(offset[2])
                if read_interval.intersects(write_node_interval):
                    write_iteration_offset = iteration_offsets.get(str(write_node.id_), None)
                    if write_iteration_offset is not None:
                        i_offsets = (
                            min(
                                write_iteration_offset.i_offsets[0],
                                read_iteration_offset.i_offsets[0] + offset[0],
                            ),
                            max(
                                write_iteration_offset.i_offsets[1],
                                read_iteration_offset.i_offsets[1] + offset[0],
                            ),
                        )
                        j_offsets = (
                            min(
                                write_iteration_offset.j_offsets[0],
                                read_iteration_offset.j_offsets[0] + offset[1],
                            ),
                            max(
                                write_iteration_offset.j_offsets[1],
                                read_iteration_offset.j_offsets[1] + offset[1],
                            ),
                        )
                    else:
                        i_offsets = (
                            read_iteration_offset.i_offsets[0] + offset[0],
                            read_iteration_offset.i_offsets[1] + offset[0],
                        )
                        j_offsets = (
                            read_iteration_offset.j_offsets[1] + offset[1],
                            read_iteration_offset.j_offsets[1] + offset[1],
                        )

                    iteration_offsets[str(write_node.id_)] = oir.CartesianIterationOffset(
                        i_offsets=i_offsets, j_offsets=j_offsets
                    )
    if write_node.id_ not in iteration_offsets:
        iteration_offsets[str(write_node.id_)] = None


def _compute_iteration_offsets(
    stencil: oir.Stencil, graph: nx.DiGraph
) -> Dict[str, Optional[oir.CartesianIterationOffset]]:
    nodes = nx.get_node_attributes(graph, "node")
    intervals = nx.get_node_attributes(graph, "interval")
    iteration_offsets: Dict[str, Optional[oir.CartesianIterationOffset]] = dict()
    from gtc.passes.oir_optimizations.utils import AccessCollector

    for id_ in reversed(list(nx.topological_sort(graph))):
        node = nodes[id_]

        access_collection = AccessCollector.apply(node)
        if any(f.name in access_collection.write_fields() for f in stencil.params):
            if node.id_ not in iteration_offsets:
                iteration_offsets[node.id_] = oir.CartesianIterationOffset(
                    i_offsets=(0, 0), j_offsets=(0, 0)
                )
            else:
                iteration_offset = iteration_offsets[node.id_]
                if iteration_offset is not None:
                    i_offsets = (
                        min(iteration_offset.i_offsets[0], 0),
                        max(iteration_offset.i_offsets[1], 0),
                    )
                    j_offsets = (
                        min(iteration_offset.j_offsets[0], 0),
                        max(iteration_offset.j_offsets[1], 0),
                    )
                    iteration_offsets[node.id_] = oir.CartesianIterationOffset(
                        i_offsets=i_offsets, j_offsets=j_offsets
                    )

        else:
            iteration_offsets.setdefault(node.id_, None)

        if id_ not in iteration_offsets or iteration_offsets[id_] is None:
            # this happens if no API output depends on this node (also not transitively) the
            # iteration space will be determined in a forward pass for completeness.
            continue

        for adj_id, _ in graph.in_edges(id_):
            _dependency_expansion_backward(
                node,
                intervals[id_],
                nodes[adj_id],
                intervals[adj_id],
                iteration_offsets,
            )

    return iteration_offsets


class OIRIterationSpaceTranslator(NodeTranslator):
    @classmethod
    def apply(cls, stencil: oir.Stencil) -> oir.Stencil:
        graph = OIRHorizontalExecutionDependencyGraphBuilder.apply(stencil)
        iteration_offsets = _compute_iteration_offsets(stencil, graph)
        return cls().visit(stencil, graph=graph, iteration_offsets=iteration_offsets)

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        iteration_offsets: Dict[str, oir.CartesianIterationOffset],
    ) -> oir.HorizontalExecution:
        return oir.HorizontalExecution(
            body=node.body,
            mask=node.mask,
            declarations=node.declarations,
            iteration_space=iteration_offsets[str(node.id_)],
        )

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        graph: nx.DiGraph,
        iteration_offsets: Dict[str, oir.CartesianIterationOffset],
    ) -> oir.VerticalLoop:
        subgraph = graph.subgraph([he.id_ for he in node.horizontal_executions])
        subgraph_nodes = nx.get_node_attributes(subgraph, "node")
        horizontal_executions = list(
            self.visit(subgraph_nodes[he], iteration_offsets=iteration_offsets)
            for he in nx.topological_sort(subgraph)
        )
        return oir.VerticalLoop(
            interval=node.interval,
            horizontal_executions=horizontal_executions,
            loop_order=node.loop_order,
            declarations=node.declarations,
        )
