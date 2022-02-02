from typing import Any, Dict, List, Set, Tuple, Union

from pydantic import root_validator, validator

from eve import Int, IntEnum, Node, Str, StrEnum
from gt4py import definitions as gt_def
from gt4py.utils import is_iterable_of
from gtc import common, oir
from gtc.common import LocNode
from gtc.dace.utils import compute_stencil_field_extents
from gtc.oir import LocalScalar, Stmt
from gtc.passes.oir_optimizations.utils import AccessCollector

from .dace.utils import get_axis_bound_diff_str, get_axis_bound_str


class Axis(StrEnum):
    I = "I"
    J = "J"
    K = "K"

    def iteration_symbol(self):
        return "__" + self.lower()

    def domain_symbol(self):
        return "__" + self.upper()

    @staticmethod
    def dims_3d():
        yield from [Axis.I, Axis.J, Axis.K]

    def to_idx(self):
        return [Axis.I, Axis.J, Axis.K].index(self)


class AxisBound(common.AxisBound):
    axis: Axis

    def __str__(self):
        return get_axis_bound_str(self, self.axis.domain_symbol())

    @classmethod
    def from_common(cls, axis, node):
        return cls(axis=axis, level=node.level, offset=node.offset)


class Decl(Node):
    name: Str
    dtype: common.DataType


class SymbolDecl(Decl):
    pass


class FieldDecl(Decl):
    shape: List[Union[Int, Str]]
    strides: List[Union[Int, Str]]
    axes: List[Axis]
    data_dims: List[Int]


class Index(Node):
    value: Union[AxisBound, Int, Str]

    @classmethod
    def from_axis(cls, axis: Axis):
        return cls(value=axis.iteration_symbol())


class IndexWithExtent(Index):
    extent: Tuple[int, int]

    @property
    def size(self):
        return self.extent[1] - self.extent[0] + 1

    def union(self, other: "Index"):
        assert self.value == other.value

        if not isinstance(other, IndexWithExtent):
            other = IndexWithExtent(value=other.value, extent=(0, 0))

        return IndexWithExtent(
            value=self.value,
            extent=(min(self.extent[0], other.extent[0]), max(self.extent[1], other.extent[1])),
        )


class Interval(Node):
    start: AxisBound
    end: AxisBound

    @property
    def size(self):

        return get_axis_bound_diff_str(
            self.end, self.start, var_name=self.start.axis.domain_symbol()
        )


class Range(Node):
    var: Str
    start: Union[Str, Int, AxisBound]
    end: Union[Str, Int, AxisBound]
    stride: Union[Str, Int] = 1

    def to_ndrange(self):
        if isinstance(self.start, AxisBound):
            start = get_axis_bound_str(self.start, self.start.axis.domain_symbol())
        else:
            start = str(self.start)
        if isinstance(self.end, AxisBound):
            end = get_axis_bound_str(self.end, self.end.axis.domain_symbol())
        else:
            end = str(self.end)
        return {self.var: f"{start}:{end}:{self.stride}"}

    @classmethod
    def from_axis_and_extent(cls, axis: Axis, extent: gt_def.Extent, stride=1):
        return cls(
            var=axis.iteration_symbol(),
            start=extent[axis.to_idx()][0],
            end=f"{axis.domain_symbol()}{extent[axis.to_idx()][1]:+d}",
            stride=stride,
        )


class GridSubset(Node):
    intervals: Dict[Axis, Union[Interval, Index]]

    def restrict(self, axis: Axis) -> "GridSubset":
        intervals = dict(self.intervals)
        intervals[axis] = Index.from_axis(axis)
        return GridSubset(intervals=intervals)

    def set_interval(self, axis: Axis, interval: Union[Interval, Index, oir.Interval]):
        if isinstance(interval, oir.Interval):
            interval = Interval(
                start=AxisBound(
                    level=interval.start.level, offset=interval.start.offset, axis=Axis.K
                ),
                end=AxisBound(level=interval.end.level, offset=interval.end.offset, axis=Axis.K),
            )
        assert interval.start.axis == axis
        intervals = dict(self.intervals)
        intervals[axis] = interval
        return GridSubset(intervals=intervals)

    @classmethod
    def from_gt4py_extent_and_interval(
        cls, extent: gt_def.Extent, interval: Union[oir.Interval, Interval, Index]
    ):
        i_interval = Interval(
            start=AxisBound(level=common.LevelMarker.START, offset=extent[0][0], axis=Axis.I),
            end=AxisBound(level=common.LevelMarker.END, offset=extent[0][1], axis=Axis.I),
        )
        j_interval = Interval(
            start=AxisBound(level=common.LevelMarker.START, offset=extent[1][0], axis=Axis.J),
            end=AxisBound(level=common.LevelMarker.END, offset=extent[1][1], axis=Axis.J),
        )
        if isinstance(interval, (Interval, oir.Interval)):
            k_interval = Interval(
                start=AxisBound(
                    level=interval.start.level, offset=interval.start.offset, axis=Axis.K
                ),
                end=AxisBound(level=interval.end.level, offset=interval.end.offset, axis=Axis.K),
            )
        else:
            assert isinstance(interval, Index)
            k_interval = interval

        return cls(intervals={Axis.I: i_interval, Axis.J: j_interval, Axis.K: k_interval})

    def to_field_extent(self):
        res = dict()
        for axis, interval in self.intervals.items():
            if isinstance(interval, Index):
                res[axis] = IndexWithExtent(value=interval.value, extent=(0, 0))
            else:
                res[axis] = interval
        return FieldExtent(values=res)


def axes_list_from_flags(flags):
    return [ax for f, ax in zip(flags, Axis.dims_3d()) if f]


def make_extent_dict(
    extent: gt_def.Extent, interval: oir.Interval, axes: List[Axis]
) -> Dict[Axis, Interval]:
    i_interval = Interval(
        start=AxisBound(level=common.LevelMarker.START, offset=extent[0][0], axis=Axis.I),
        end=AxisBound(level=common.LevelMarker.END, offset=extent[0][1], axis=Axis.I),
    )
    j_interval = Interval(
        start=AxisBound(level=common.LevelMarker.START, offset=extent[1][0], axis=Axis.J),
        end=AxisBound(level=common.LevelMarker.END, offset=extent[1][1], axis=Axis.J),
    )
    if isinstance(interval, Index):
        k_interval = interval
    else:
        k_interval = Interval(
            start=AxisBound(level=interval.start.level, offset=interval.start.offset, axis=Axis.K),
            end=AxisBound(level=interval.end.level, offset=interval.end.offset, axis=Axis.K),
        )
    res = {Axis.I: i_interval, Axis.J: j_interval, Axis.K: k_interval}
    return {axis: res[axis] for axis in axes}


class FieldExtent(Node):
    values: Dict[Axis, Union[Interval, IndexWithExtent]]
    dynamic_access: bool = False
    dynamic_offsets: List[Axis] = []
    data_dims: List[Int]

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [Axis.I, Axis.J, Axis.K][item]
        return self.values[item]

    @property
    def shape(self):
        res = []
        for extent in iter(self):
            res.append(extent.size)
        return tuple(res)

    def __iter__(self):
        for axis in Axis.dims_3d():
            if axis in self.values:
                yield self.values[axis]

    def items(self):
        for axis in Axis.dims_3d():
            if axis in self.values:
                yield axis, self.values

    @property
    def axes(self):
        for axis in Axis.dims_3d():
            if axis in self.values:
                yield axis

    @property
    def needs_dynamic_edge(self) -> bool:
        return self.dynamic_access or len(self.dynamic_offsets) > 0


class ComputationNode(Node):
    read_extents: Dict[str, FieldExtent]
    write_extents: Dict[str, FieldExtent]


class IterationNode(Node):
    grid_subset: GridSubset


class NestedSDFGNode(ComputationNode):
    field_extents: Dict[str, FieldExtent]
    declarations: List[Decl]


class Tasklet(ComputationNode, LocNode):
    stmts: List[Union[LocalScalar, Stmt]]


class DomainMap(ComputationNode, IterationNode):
    index_range: Range
    computations: List[Union[Tasklet, "DomainMap"]]


class ComputationState(IterationNode):
    computations: List[Union[Tasklet, DomainMap]]


class DomainLoop(IterationNode):
    index_range: Range
    loop_states: List[ComputationState]


class StateMachine(NestedSDFGNode):
    states: List[Union[DomainLoop, ComputationState]]


def make_read_extents(node: common.Node, grid_subset: GridSubset, block_extents, context_extents):
    in_extents = compute_stencil_field_extents(
        node, block_extents=block_extents, read=True, write=False
    )
    read_extents = dict()
    dynamic_fields = {
        acc.name
        for mask in node.iter_tree().if_isinstance(oir.MaskStmt)
        for stmt in mask.body
        for acc in stmt.left.iter_tree().if_isinstance(oir.FieldAccess)
        if isinstance(stmt, oir.AssignStmt)
    }
    for field, extent in in_extents.items():
        res = dict()

        for axis in context_extents[field].axes:
            axis_idx = axis.to_idx()
            grid_int = grid_subset.intervals[axis]
            if isinstance(grid_int, Interval):
                if axis == Axis.K:
                    res[axis] = Interval(
                        start=AxisBound(
                            axis=axis,
                            level=grid_int.start.level,
                            offset=grid_int.start.offset + extent[axis_idx][0],
                        ),
                        end=AxisBound(
                            axis=axis,
                            level=grid_int.end.level,
                            offset=grid_int.end.offset + extent[axis_idx][1],
                        ),
                    )
                else:
                    res[axis] = Interval(
                        start=AxisBound(
                            axis=axis,
                            level=grid_int.start.level,
                            offset=extent[axis_idx][0],
                        ),
                        end=AxisBound(
                            axis=axis,
                            level=grid_int.end.level,
                            offset=extent[axis_idx][1],
                        ),
                    )
            else:
                res[axis] = IndexWithExtent(value=grid_int.value, extent=extent[axis_idx])
        read_extents[field] = FieldExtent(
            values=res,
            dynamic_access=(field in dynamic_fields),
            data_dims=context_extents[field].data_dims,
        )
    return read_extents


def make_write_extents(node: common.Node, grid_subset: GridSubset, block_extents, context_extents):
    write_fields = AccessCollector.apply(node).write_fields()
    write_extents = dict()
    dynamic_fields = {
        acc.name
        for mask in node.iter_tree().if_isinstance(oir.MaskStmt)
        for stmt in mask.body
        for acc in stmt.left.iter_tree().if_isinstance(oir.FieldAccess)
        if isinstance(stmt, oir.AssignStmt)
    }
    for field in write_fields:
        res = dict()
        for axis in context_extents[field].axes:
            interval = grid_subset.intervals[axis]
            if isinstance(interval, Index):
                res[axis] = IndexWithExtent(value=interval.value, extent=(0, 0))
            else:
                res[axis] = interval
        write_extents[field] = FieldExtent(
            values=res,
            dynamic_access=(field in dynamic_fields),
            data_dims=context_extents[field].data_dims,
        )
    return write_extents
