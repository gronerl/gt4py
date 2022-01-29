from typing import Dict, List, Set, Tuple, Union

from eve import Int, IntEnum, Node, Str, StrEnum
from gt4py import definitions as gt_def
from gtc import common, oir
from gtc.common import LocNode
from gtc.dace.utils import compute_stencil_field_extents
from gtc.oir import LocalScalar, Stmt

from .dace.utils import get_axis_bound_diff_str, get_axis_bound_str


class Axis(StrEnum):
    I = "I"
    J = "J"
    K = "K"

    def iteration_symbol(self):
        return "__" + self.lower()

    def domain_symbol(self):
        return "__" + self.upper()


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
    shape: Tuple[Union[Int, Str], Union[Int, Str], Union[Int, Str]]
    strides: Tuple[Union[Int, Str], Union[Int, Str], Union[Int, Str]]


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
            start = get_axis_bound_str(self.start, "__" + self.start.axis.domain_symbol())
        else:
            start = str(self.start)
        if isinstance(self.end, AxisBound):
            end = get_axis_bound_str(self.end, "__" + self.end.axis.domain_symbol())
        else:
            end = str(self.end)
        return {self.var: f"{start}:{end}:{self.stride}"}

    @classmethod
    def from_axis_and_extent(cls, axis: Axis, extent: gt_def.Extent, stride=1):
        return cls(
            var=axis.iteration_symbol(),
            start=extent[0][0],
            end=f"{axis.domain_symbol()}{extent[0][1]:+d}",
            stride=stride,
        )


class GridSubset(Node):
    intervals: Dict[Axis, Union[Interval, Index]]

    def restrict(self, axis: Axis) -> "GridSubset":
        intervals = dict(self.intervals)
        intervals[axis] = Index.from_axis(axis)
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


class FieldExtent(Node):
    values: Dict[Axis, Union[Interval, IndexWithExtent]]
    dynamic_volume: bool = False
    dynamic_offset: bool = False

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [Axis.I, Axis.J, Axis.K][item]
        return self.values[item]

    @classmethod
    def from_gt4py_extent_and_interval(cls, extent: gt_def.Extent, interval: oir.Interval):
        i_interval = Interval(
            start=AxisBound(level=common.LevelMarker.START, offset=extent[0][0], axis=Axis.I),
            end=AxisBound(level=common.LevelMarker.END, offset=extent[0][1], axis=Axis.I),
        )
        j_interval = Interval(
            start=AxisBound(level=common.LevelMarker.START, offset=extent[1][0], axis=Axis.J),
            end=AxisBound(level=common.LevelMarker.END, offset=extent[1][1], axis=Axis.J),
        )
        k_interval = Interval(
            start=AxisBound(level=interval.start.level, offset=interval.start.offset, axis=Axis.K),
            end=AxisBound(level=interval.end.level, offset=interval.end.offset, axis=Axis.K),
        )
        return cls(values={Axis.I: i_interval, Axis.J: j_interval, Axis.K: k_interval})

    @property
    def shape(self):
        res = []
        for axis in [Axis.I, Axis.J, Axis.K]:
            extent = self.values[axis]
            res.append(extent.size)
        return tuple(res)

    def __iter__(self):
        for axis in [Axis.I, Axis.J, Axis.K]:
            yield self.values[axis]


class ComputationNode(Node):
    read_extents: Dict[str, FieldExtent]
    out_fields: Set[str]


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


def make_read_extents(node: common.Node, grid_subset: GridSubset, block_extents):
    in_extents = compute_stencil_field_extents(
        node, block_extents=block_extents, read=True, write=False
    )
    read_extents = dict()
    for field, extent in in_extents.items():
        res = dict()

        for axis, axis_idx in [(Axis.I, 0), (Axis.J, 1), (Axis.K, 2)]:
            grid_int = grid_subset.intervals[axis]
            if isinstance(grid_int, Interval):
                res[axis] = Interval(
                    start=AxisBound(
                        axis=axis, level=common.LevelMarker.START, offset=extent[axis_idx][0]
                    ),
                    end=AxisBound(
                        axis=axis, level=common.LevelMarker.END, offset=extent[axis_idx][1]
                    ),
                )
            else:
                res[axis] = IndexWithExtent(value=grid_int.value, extent=extent[axis_idx])
        read_extents[field] = FieldExtent(values=res)
    return read_extents
