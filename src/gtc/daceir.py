from typing import Dict, List, Tuple, Union

from eve import Int, Node, Str, StrEnum
from gt4py import definitions as gt_def
from gtc import common, oir
from gtc.common import LocNode
from gtc.oir import LocalScalar, Stmt

from .dace.utils import get_axis_bound_diff_str, get_axis_bound_str


class Axis(StrEnum):
    I = "I"  # noqa: E741 ambiguous variable name 'I'
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


class IndexWithExtent(Node):
    value: Union[AxisBound, Int, Str]
    extent: Tuple[int, int]
    variable: bool = False

    @classmethod
    def from_axis(cls, axis: Axis, extent=(0, 0)):
        return cls(value=axis.iteration_symbol(), extent=extent)

    @property
    def size(self):
        return self.extent[1] - self.extent[0] + 1

    def union(self, other: "IndexWithExtent"):
        assert self.value == other.value

        return IndexWithExtent(
            value=self.value,
            extent=(min(self.extent[0], other.extent[0]), max(self.extent[1], other.extent[1])),
            variable=self.variable or other.variable,
        )

    @property
    def idx_range(self):
        return f"{self.value}{self.extent[0]:+d}", f"{self.value}{self.extent[1] + 1:+d}"


class Interval(Node):
    start: AxisBound
    end: AxisBound
    variable: bool = False

    @property
    def size(self):
        return get_axis_bound_diff_str(
            self.end, self.start, var_name=self.start.axis.domain_symbol()
        )

    @classmethod
    def union(cls, first, second):
        return cls(
            start=min(first.start, second.start),
            end=max(first.end, second.end),
            variable=first.variable or second.variable,
        )

    @property
    def idx_range(self):
        return str(self.start), str(self.end)


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
    intervals: Dict[Axis, Union[Interval, IndexWithExtent]]

    def __iter__(self):
        for axis in Axis.dims_3d():
            if axis in self.intervals:
                yield self.intervals[axis]

    def items(self):
        for axis in Axis.dims_3d():
            if axis in self.intervals:
                yield axis, self.intervals[axis]

    @property
    def shape(self):
        return tuple(interval.size for _, interval in self.items())

    def restricted_to_index(self, axis: Axis, extent=(0, 0)) -> "GridSubset":
        intervals = dict(self.intervals)
        intervals[axis] = IndexWithExtent.from_axis(axis, extent=extent)
        return GridSubset(intervals=intervals)

    def set_interval(
        self, axis: Axis, interval: Union[Interval, IndexWithExtent, oir.Interval]
    ) -> "GridSubset":
        if isinstance(interval, oir.Interval):
            interval = Interval(
                start=AxisBound(
                    level=interval.start.level, offset=interval.start.offset, axis=Axis.K
                ),
                end=AxisBound(level=interval.end.level, offset=interval.end.offset, axis=Axis.K),
            )
        elif isinstance(interval, Interval):
            assert interval.start.axis == axis
        intervals = dict(self.intervals)
        intervals[axis] = interval
        return GridSubset(intervals=intervals)

    @classmethod
    def from_gt4py_extent(cls, extent: gt_def.Extent):
        i_interval = Interval(
            start=AxisBound(level=common.LevelMarker.START, offset=extent[0][0], axis=Axis.I),
            end=AxisBound(level=common.LevelMarker.END, offset=extent[0][1], axis=Axis.I),
        )
        j_interval = Interval(
            start=AxisBound(level=common.LevelMarker.START, offset=extent[1][0], axis=Axis.J),
            end=AxisBound(level=common.LevelMarker.END, offset=extent[1][1], axis=Axis.J),
        )

        return cls(intervals={Axis.I: i_interval, Axis.J: j_interval})

    @classmethod
    def from_interval(cls, interval: Union[oir.Interval, Interval, IndexWithExtent], axis: Axis):
        if isinstance(interval, (Interval, oir.Interval)):
            res_interval = Interval(
                start=AxisBound(
                    level=interval.start.level, offset=interval.start.offset, axis=Axis.K
                ),
                end=AxisBound(level=interval.end.level, offset=interval.end.offset, axis=Axis.K),
            )
        else:
            assert isinstance(interval, IndexWithExtent)
            res_interval = interval

        return cls(intervals={axis: res_interval})

    def axes(self):
        for axis in Axis.dims_3d():
            if axis in self.intervals:
                yield axis

    @classmethod
    def full_domain(cls, axes=None):
        if axes is None:
            axes = Axis.dims_3d()
        res_subsets = dict()
        for axis in axes:
            res_subsets[axis] = Interval(
                start=AxisBound(axis=axis, level=common.LevelMarker.START, offset=0),
                end=AxisBound(axis=axis, level=common.LevelMarker.END, offset=0),
            )
        return GridSubset(intervals=res_subsets)


class FieldAccessInfo(Node):
    grid_subset: GridSubset
    dynamic_access: bool = False
    variable_offset_axes: List[Axis] = []

    @property
    def is_dynamic(self) -> bool:
        return self.dynamic_access or len(self.variable_offset_axes) > 0

    def axes(self):
        yield from self.grid_subset.axes()

    @property
    def shape(self):
        return self.grid_subset.shape

    def apply_iteration(self, grid_subset: GridSubset):
        res_intervals = dict(self.grid_subset.intervals)
        for axis, field_interval in self.grid_subset.intervals.items():
            if axis in grid_subset.intervals:
                grid_interval = grid_subset.intervals[axis]
                assert isinstance(field_interval, IndexWithExtent)
                if isinstance(grid_interval, Interval):
                    extent = field_interval.extent
                    res_intervals[axis] = Interval(
                        start=AxisBound(
                            axis=axis,
                            level=grid_interval.start.level,
                            offset=grid_interval.start.offset + extent[0],
                        ),
                        end=AxisBound(
                            axis=axis,
                            level=grid_interval.end.level,
                            offset=grid_interval.end.offset + extent[1],
                        ),
                        variable=field_interval.variable,
                    )
                else:
                    assert field_interval.value == grid_interval.value
                    extent = (
                        min(field_interval.extent) + grid_interval.extent[0],
                        max(field_interval.extent) + grid_interval.extent[1],
                    )
                    res_intervals[axis] = IndexWithExtent(
                        value=field_interval.value, extent=extent, variable=field_interval.variable
                    )
        return FieldAccessInfo(
            grid_subset=GridSubset(intervals=res_intervals),
            dynamic_access=self.dynamic_access,
            variable_offset_axes=self.variable_offset_axes,
        )

    def union(self, other: "FieldAccessInfo"):
        assert list(self.axes()) == list(other.axes())
        intervals = dict()
        for axis in self.axes():
            interval1 = self.grid_subset.intervals[axis]
            interval2 = other.grid_subset.intervals[axis]
            if isinstance(interval1, Interval) and isinstance(interval2, Interval):
                intervals[axis] = Interval.union(interval1, interval2)
            else:
                assert isinstance(interval1, IndexWithExtent) and isinstance(
                    interval2, IndexWithExtent
                )
                intervals[axis] = interval1.union(interval2)
        variable_offset_axes = [
            axis
            for axis in Axis.dims_3d()
            if axis in self.variable_offset_axes or axis in other.variable_offset_axes
        ]
        return FieldAccessInfo(
            grid_subset=GridSubset(intervals=intervals),
            dynamic_access=self.dynamic_access or other.dynamic_access,
            variable_offset_axes=variable_offset_axes,
        )

    def clamp_full_axis(self, axis):
        grid_subset = GridSubset(intervals=self.grid_subset.intervals)
        interval = self.grid_subset.intervals[axis]
        full_interval = Interval(
            start=AxisBound(level=common.LevelMarker.START, offset=0, axis=axis),
            end=AxisBound(level=common.LevelMarker.END, offset=0, axis=axis),
        )
        if isinstance(interval, Interval):
            interval_union = Interval.union(interval, full_interval)
            grid_subset.intervals[axis] = interval_union
        else:
            grid_subset.intervals[axis] = full_interval
            grid_subset = grid_subset.set_interval(axis, full_interval)
        return FieldAccessInfo(grid_subset=grid_subset)


class FieldDecl(Node):
    name: Str
    dtype: common.DataType
    strides: List[Union[Int, Str]]
    data_dims: List[Int]
    access_info: FieldAccessInfo

    @property
    def shape(self):
        ijk_shape = self.access_info.grid_subset.shape
        return ijk_shape + tuple(self.data_dims)

    def axes(self):
        yield from self.access_info.grid_subset.axes()

    @property
    def is_dynamic(self) -> bool:
        return self.access_info.is_dynamic

    def with_set_access_info(self, access_info: FieldAccessInfo) -> "FieldDecl":
        return FieldDecl(
            name=self.name,
            dtype=self.dtype,
            strides=self.strides,
            data_dims=self.data_dims,
            access_info=access_info,
        )


class ComputationNode(Node):
    # mapping connector names to tuple of field name and access info
    read_accesses: Dict[str, FieldAccessInfo]
    write_accesses: Dict[str, FieldAccessInfo]


class IterationNode(Node):
    grid_subset: GridSubset


class NestedSDFGNode(ComputationNode):
    field_decls: Dict[str, FieldDecl]
    symbols: Dict[str, common.DataType]


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
