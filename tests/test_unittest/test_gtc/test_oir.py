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

import functools
import itertools
from types import SimpleNamespace

import hypothesis as hyp
import pytest
from hypothesis import strategies as hyp_st
from pydantic.error_wrappers import ValidationError

from gtc.common import (
    ArithmeticOperator,
    CartesianOffset,
    ComparisonOperator,
    DataType,
    ExprKind,
    LevelMarker,
    LoopOrder,
)
from gtc.gtir_to_oir import (
    OIRHorizontalExecutionDependencyGraphBuilder,
    _compute_iteration_offsets,
    _dependency_expansion_backward,
)
from gtc.oir import (
    AssignStmt,
    AxisBound,
    BinaryOp,
    CartesianIterationOffset,
    Expr,
    FieldAccess,
    FieldDecl,
    HorizontalExecution,
    Interval,
    IntervalMapping,
    Literal,
    Stencil,
    Temporary,
    VerticalLoop,
)


A_ARITHMETIC_TYPE = DataType.INT32


@functools.lru_cache(maxsize=None)
def get_instance(value: int) -> SimpleNamespace:
    return SimpleNamespace(value=value)


@hyp_st.composite
def intervals_strategy(draw):
    length = draw(hyp_st.integers(0, 5))
    intervals = []
    for _ in range(length):
        level1 = draw(hyp_st.sampled_from([LevelMarker.START, LevelMarker.END]))
        offset1 = draw(hyp_st.integers(-5, 5))
        bound1 = AxisBound(level=level1, offset=offset1)

        level2 = draw(hyp_st.sampled_from([LevelMarker.START, LevelMarker.END]))
        offset2 = draw(
            hyp_st.integers(-5, 5).filter(lambda x: x != offset1 if level1 == level2 else True)
        )
        bound2 = AxisBound(level=level2, offset=offset2)

        intervals.append(Interval(start=min(bound1, bound2), end=max(bound1, bound2)))
    return intervals


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    kind: ExprKind = ExprKind.FIELD


def test_no_horizontal_offset_allowed():
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        AssignStmt(
            left=FieldAccess(
                name="foo", dtype=A_ARITHMETIC_TYPE, offset=CartesianOffset(i=1, j=0, k=0)
            ),
            right=DummyExpr(dtype=A_ARITHMETIC_TYPE),
        ),


def test_mask_must_be_bool():
    with pytest.raises(ValidationError, match=r".*must be.* bool.*"):
        HorizontalExecution(body=[], mask=DummyExpr(dtype=A_ARITHMETIC_TYPE)),


EQUAL_AXISBOUNDS = [
    (AxisBound.start(), AxisBound.start()),
    (AxisBound.end(), AxisBound.end()),
    (AxisBound.from_end(-1), AxisBound.from_end(-1)),
]
LESS_AXISBOUNDS = [
    (AxisBound.start(), AxisBound.end()),
    (AxisBound.start(), AxisBound.from_start(1)),
    (AxisBound.from_end(-1), AxisBound.end()),
    (AxisBound.from_start(1), AxisBound.from_end(-1)),
]
GREATER_AXISBOUNDS = [
    (AxisBound.end(), AxisBound.start()),
    (AxisBound.from_start(1), AxisBound.start()),
    (AxisBound.end(), AxisBound.from_end(-1)),
    (AxisBound.from_end(-1), AxisBound.from_start(1)),
]


class TestAxisBoundsComparison:
    @pytest.mark.parametrize(["lhs", "rhs"], EQUAL_AXISBOUNDS)
    def test_eq_true(self, lhs, rhs):
        res1 = lhs == rhs
        assert isinstance(res1, bool)
        assert res1

        res2 = rhs == lhs
        assert isinstance(res2, bool)
        assert res2

    @pytest.mark.parametrize(
        ["lhs", "rhs"],
        LESS_AXISBOUNDS + GREATER_AXISBOUNDS,
    )
    def test_eq_false(self, lhs, rhs):
        res1 = lhs == rhs
        assert isinstance(res1, bool)
        assert not res1

        res2 = rhs == lhs
        assert isinstance(res2, bool)
        assert not res2

    @pytest.mark.parametrize(["lhs", "rhs"], LESS_AXISBOUNDS)
    def test_lt_true(self, lhs, rhs):
        res = lhs < rhs
        assert isinstance(res, bool)
        assert res

    @pytest.mark.parametrize(
        ["lhs", "rhs"],
        GREATER_AXISBOUNDS + EQUAL_AXISBOUNDS,
    )
    def test_lt_false(self, lhs, rhs):
        res = lhs < rhs
        assert isinstance(res, bool)
        assert not res

    @pytest.mark.parametrize(["lhs", "rhs"], GREATER_AXISBOUNDS)
    def test_gt_true(self, lhs, rhs):
        res = lhs > rhs
        assert isinstance(res, bool)
        assert res

    @pytest.mark.parametrize(["lhs", "rhs"], LESS_AXISBOUNDS + EQUAL_AXISBOUNDS)
    def test_gt_false(self, lhs, rhs):
        res = lhs > rhs
        assert isinstance(res, bool)
        assert not res

    @pytest.mark.parametrize(["lhs", "rhs"], LESS_AXISBOUNDS + EQUAL_AXISBOUNDS)
    def test_le_true(self, lhs, rhs):
        res = lhs <= rhs
        assert isinstance(res, bool)
        assert res

    @pytest.mark.parametrize(["lhs", "rhs"], GREATER_AXISBOUNDS)
    def test_le_false(self, lhs, rhs):
        res = lhs <= rhs
        assert isinstance(res, bool)
        assert not res

    @pytest.mark.parametrize(["lhs", "rhs"], GREATER_AXISBOUNDS + EQUAL_AXISBOUNDS)
    def test_ge_true(self, lhs, rhs):
        res = lhs >= rhs
        assert isinstance(res, bool)
        assert res

    @pytest.mark.parametrize(["lhs", "rhs"], LESS_AXISBOUNDS)
    def test_ge_false(self, lhs, rhs):
        res = lhs >= rhs
        assert isinstance(res, bool)
        assert not res


COVER_INTERVALS = [
    (
        Interval(start=AxisBound.start(), end=AxisBound.end()),
        Interval(start=AxisBound.start(), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.end()),
        Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.end()),
        Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.end()),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-1)),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
        Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(4)),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(3)),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
        Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(3)),
    ),
]
SUBSET_INTERVALS = [
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
        Interval(start=AxisBound.start(), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
        Interval(start=AxisBound.start(), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-1)),
        Interval(start=AxisBound.start(), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(4)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(3)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
    ),
    (
        Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(3)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
    ),
]
DISJOINT_INTERVALS = [
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
        Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_start(3)),
        Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_start(3)),
        Interval(start=AxisBound.from_end(1), end=AxisBound.from_end(2)),
    ),
]
OVERLAP_INTERVALS = [
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
        Interval(start=AxisBound.from_end(-2), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
        Interval(start=AxisBound.start(), end=AxisBound.from_start(2)),
    ),
]


class TestIntervalOperations:
    @pytest.mark.parametrize(["lhs", "rhs"], COVER_INTERVALS)
    def test_covers_true(self, lhs, rhs):
        res = lhs.covers(rhs)
        assert isinstance(res, bool)
        assert res

    @pytest.mark.parametrize(
        ["lhs", "rhs"], SUBSET_INTERVALS + OVERLAP_INTERVALS + DISJOINT_INTERVALS
    )
    def test_covers_false(self, lhs, rhs):
        res = lhs.covers(rhs)
        assert isinstance(res, bool)
        assert not res

    @pytest.mark.parametrize(["lhs", "rhs"], COVER_INTERVALS + SUBSET_INTERVALS + OVERLAP_INTERVALS)
    def test_intersects_true(self, lhs, rhs):
        res1 = lhs.intersects(rhs)
        assert isinstance(res1, bool)
        assert res1

        res2 = lhs.intersects(rhs)
        assert isinstance(res2, bool)
        assert res2

    @pytest.mark.parametrize(["lhs", "rhs"], DISJOINT_INTERVALS)
    def test_intersects_false(self, lhs, rhs):
        res1 = lhs.intersects(rhs)
        assert isinstance(res1, bool)
        assert not res1

        res2 = rhs.intersects(lhs)
        assert isinstance(res2, bool)
        assert not res2


class TestIntervalMapping:
    @staticmethod
    def assert_consistency(imap: IntervalMapping):
        assert len(imap.interval_starts) == len(imap.interval_ends)
        assert len(imap.interval_starts) == len(imap.values)
        for i in range(len(imap.interval_starts) - 1):
            assert imap.interval_starts[i] < imap.interval_starts[i + 1]
            assert imap.interval_ends[i] < imap.interval_ends[i + 1]
            assert imap.interval_ends[i] <= imap.interval_starts[i + 1]
            if imap.interval_ends[i] == imap.interval_starts[i + 1]:
                assert imap.values[i] is not imap.values[i + 1]

        for start, end in zip(imap.interval_starts, imap.interval_ends):
            assert start < end

    @pytest.mark.parametrize(
        ["intervals", "starts", "ends"],
        [
            ([], [], []),
            (
                [Interval(start=AxisBound.start(), end=AxisBound.end())],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.end()),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_start(2)],
                [AxisBound.from_start(1), AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-1)),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_end(-2)),
                ],
                [AxisBound.start(), AxisBound.from_start(2), AxisBound.from_end(-1)],
                [AxisBound.from_start(1), AxisBound.from_end(-2), AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-2)),
                ],
                [AxisBound.start(), AxisBound.from_end(-1)],
                [AxisBound.from_end(-2), AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_end(-1)),
                ],
                [AxisBound.start(), AxisBound.from_start(2)],
                [AxisBound.from_start(1), AxisBound.end()],
            ),
        ],
    )
    def test_setitem_same_value(self, intervals, starts, ends):
        # if all values are the same (same instance), the behavior is the same as for a IntervalSet.
        imap = IntervalMapping()
        for interval in intervals:
            imap[interval] = 0
            self.assert_consistency(imap)

        assert len(starts) == len(imap.interval_starts)
        for expected, observed in zip(starts, imap.interval_starts):
            assert observed == expected

        assert len(ends) == len(imap.interval_ends)
        for expected, observed in zip(ends, imap.interval_ends):
            assert observed == expected

    @hyp.given(intervals_strategy())
    def test_setitem_same_value_hypothesis(self, intervals):
        imap = IntervalMapping()
        for interval in intervals:
            imap[interval] = 0
            self.assert_consistency(imap)

        for permutation in itertools.permutations(intervals):
            other_imap = IntervalMapping()
            for interval in permutation:
                other_imap[interval] = 0
                self.assert_consistency(other_imap)
            assert imap.interval_starts == other_imap.interval_starts
            assert imap.interval_ends == other_imap.interval_ends

    @pytest.mark.parametrize(
        ["intervals", "starts", "ends", "values"],
        [
            ([], [], [], []),
            (
                [Interval(start=AxisBound.start(), end=AxisBound.end())],
                [AxisBound.start()],
                [AxisBound.end()],
                [0],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_start(1)],
                [AxisBound.from_start(1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.end()),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_end(-1)],
                [AxisBound.from_end(-1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_end(-1)],
                [AxisBound.from_end(-1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_start(1)],
                [AxisBound.from_start(1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_start(2)],
                [AxisBound.from_start(1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_start(1)],
                [AxisBound.from_start(1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-1)),
                ],
                [AxisBound.start(), AxisBound.from_start(1), AxisBound.from_end(-1)],
                [AxisBound.from_start(1), AxisBound.from_end(-1), AxisBound.end()],
                [0, 2, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_end(-2)),
                ],
                [AxisBound.start(), AxisBound.from_start(2), AxisBound.from_end(-1)],
                [AxisBound.from_start(1), AxisBound.from_end(-2), AxisBound.end()],
                [0, 2, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-2)),
                ],
                [AxisBound.start(), AxisBound.from_start(1), AxisBound.from_end(-1)],
                [AxisBound.from_start(1), AxisBound.from_end(-2), AxisBound.end()],
                [0, 2, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_end(-1)),
                ],
                [AxisBound.start(), AxisBound.from_start(2), AxisBound.from_end(-1)],
                [AxisBound.from_start(1), AxisBound.from_end(-1), AxisBound.end()],
                [0, 2, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(3)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(2)),
                ],
                [AxisBound.start(), AxisBound.from_start(1), AxisBound.from_start(2)],
                [AxisBound.from_start(1), AxisBound.from_start(2), AxisBound.from_start(3)],
                [0, 1, 0],
            ),
            (
                [
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(3)),
                ],
                [AxisBound.start()],
                [AxisBound.from_start(3)],
                [1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                ],
                [AxisBound.start(), AxisBound.from_start(1)],
                [AxisBound.from_start(1), AxisBound.from_start(2)],
                [1, 0],
            ),
        ],
    )
    def test_setitem_different_value(self, intervals, starts, ends, values):
        imap = IntervalMapping()
        ctr = 0
        for interval in intervals:
            imap[interval] = get_instance(ctr)
            self.assert_consistency(imap)
            ctr = ctr + 1

        assert len(imap.interval_starts) == len(starts)
        assert len(imap.interval_ends) == len(ends)
        assert len(imap.values) == len(values)
        for i, (start, end, value) in enumerate(
            zip(imap.interval_starts, imap.interval_ends, imap.values)
        ):
            assert start == starts[i]
            assert end == ends[i]
            assert value is get_instance(values[i])

    @hyp.given(intervals_strategy())
    def test_setitem_different_value_hypothesis(self, intervals):
        ctr = 0
        imap = IntervalMapping()
        for interval in intervals:
            imap[interval] = get_instance(ctr)
            self.assert_consistency(imap)
            ctr += 1
        for permutation in itertools.permutations(intervals):
            other_imap = IntervalMapping()
            for interval in permutation:
                other_imap[interval] = get_instance(ctr)
                self.assert_consistency(other_imap)
                ctr += 1

            for start, end, value in zip(
                other_imap.interval_starts, other_imap.interval_ends, other_imap.values
            ):
                if start == permutation[-1].start:
                    assert end == permutation[-1].end
                    assert value is get_instance(ctr - 1)
                    break

    @pytest.mark.parametrize(
        ["interval", "values"],
        [
            (Interval(start=AxisBound.from_start(-1), end=AxisBound.from_end(1)), [0, 1]),
            (Interval(start=AxisBound.from_start(-1), end=AxisBound.from_start(3)), [0]),
            (Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-1)), [0, 1]),
            (Interval(start=AxisBound.from_start(2), end=AxisBound.from_end(-2)), []),
        ],
    )
    def test_getitem_different_value(self, interval, values):
        imap = IntervalMapping()
        imap[Interval(start=AxisBound.start(), end=AxisBound.from_start(2))] = get_instance(0)
        imap[Interval(start=AxisBound.from_end(-2), end=AxisBound.end())] = get_instance(1)
        res = imap[interval]
        assert isinstance(res, list)
        assert len(res) == len(values)
        for observed, expected in zip(res, values):
            assert observed is get_instance(expected)


class TestGraphBuilder:
    def test_write_write_dependency(self):
        raise NotImplementedError

    def test_write_write_no_overlap(self):
        raise NotImplementedError

    def test_write_write_partially_shadowed_by_read(self):
        raise NotImplementedError

    def test_write_write_shadowed_by_read(self):
        raise NotImplementedError

    def test_read_write_dependency(self):
        raise NotImplementedError

    def test_read_write_no_overlap(self):
        raise NotImplementedError

    def test_independent_nodes(self):
        raise NotImplementedError

    def test_vertical_loop_ignored(self):
        raise NotImplementedError

    def test_no_dependencies_no_edges(self):
        raise NotImplementedError


class TestIterationOffsetComputationUtils:
    class TestDependencyExpansion:
        def test_k_offset_not_considered_shadowed(self):
            write_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.BOOL),
                    )
                ],
                mask=None,
                iteration_space=None,
            )
            read_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="2.0", dtype=DataType.FLOAT64),
                    ),
                    AssignStmt(
                        left=FieldAccess(name="out", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=1)),
                    ),
                ],
                mask=BinaryOp(
                    op=ComparisonOperator.EQ,
                    left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                    right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    dtype=DataType.BOOL,
                ),
                iteration_space=None,
            )
            iteration_offsets = {
                read_node.id_: CartesianIterationOffset(i_offsets=(0, 0), j_offsets=(0, 0))
            }
            _dependency_expansion_backward(
                read_node,
                Interval(start=AxisBound.start(), end=AxisBound.end()),
                write_node,
                Interval(start=AxisBound.start(), end=AxisBound.end()),
                iteration_offsets,
            )
            assert write_node.id_ in iteration_offsets
            assert iteration_offsets[write_node.id_].i_offsets == (0, 0)
            assert iteration_offsets[write_node.id_].j_offsets == (0, 0)

        def test_mask_before_shadowed_write_not_none(self):
            write_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.BOOL),
                    )
                ],
                mask=None,
                iteration_space=None,
            )
            read_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="2.0", dtype=DataType.FLOAT64),
                    ),
                    AssignStmt(
                        left=FieldAccess(name="out", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                    ),
                ],
                mask=BinaryOp(
                    op=ComparisonOperator.EQ,
                    left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                    right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    dtype=DataType.BOOL,
                ),
                iteration_space=None,
            )
            iteration_offsets = {
                read_node.id_: CartesianIterationOffset(i_offsets=(0, 0), j_offsets=(0, 0))
            }
            _dependency_expansion_backward(
                read_node,
                Interval(start=AxisBound.start(), end=AxisBound.end()),
                write_node,
                Interval(start=AxisBound.start(), end=AxisBound.end()),
                iteration_offsets,
            )
            assert write_node.id_ in iteration_offsets
            assert iteration_offsets[write_node.id_].i_offsets == (0, 0)
            assert iteration_offsets[write_node.id_].j_offsets == (0, 0)

        def test_write_before_read_nooffset_none(self):
            write_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.BOOL),
                    )
                ],
                mask=None,
                iteration_space=None,
            )
            read_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="2.0", dtype=DataType.FLOAT64),
                    ),
                    AssignStmt(
                        left=FieldAccess(name="out", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                    ),
                ],
                mask=None,
                iteration_space=None,
            )
            iteration_offsets = {
                read_node.id_: CartesianIterationOffset(i_offsets=(0, 0), j_offsets=(0, 0))
            }
            _dependency_expansion_backward(
                read_node,
                Interval(start=AxisBound.start(), end=AxisBound.end()),
                write_node,
                Interval(start=AxisBound.start(), end=AxisBound.end()),
                iteration_offsets,
            )
            assert write_node.id_ in iteration_offsets
            assert iteration_offsets[write_node.id_] is None

        def test_masked_field(self):
            write_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.BOOL),
                    )
                ],
                mask=None,
                iteration_space=None,
            )
            read_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="out", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                mask=FieldAccess(
                    name="tmp", offset=CartesianOffset(i=0, j=1, k=0), dtype=DataType.BOOL
                ),
                iteration_space=None,
            )
            iteration_offsets = {
                read_node.id_: CartesianIterationOffset(i_offsets=(0, 0), j_offsets=(0, 0))
            }
            _dependency_expansion_backward(
                read_node,
                Interval(start=AxisBound.start(), end=AxisBound.end()),
                write_node,
                Interval(start=AxisBound.start(), end=AxisBound.end()),
                iteration_offsets,
            )
            assert write_node.id_ in iteration_offsets
            assert iteration_offsets[write_node.id_].i_offsets == (0, 0)
            assert iteration_offsets[write_node.id_].j_offsets == (1, 1)

        @pytest.mark.parametrize(
            [
                "read_offsets_1",
                "read_offsets_2",
                "iteration_offset",
                "write_node_interval",
                "read_node_interval",
            ],
            [
                [[(0, 0, 0)], [], ((0, 0), (0, 0)), None, None],
                [[(1, 0, 0)], [], ((1, 1), (0, 0)), None, None],
                [[(0, 1, 0)], [], ((0, 0), (1, 1)), None, None],
                [[(0, 0, 1)], [], ((0, 0), (0, 0)), None, None],
                [[(-3, -7, 0)], [], ((-3, -3), (-7, -7)), None, None],
                [
                    [(-3, -7, 0), (-2, -6, 0)],
                    [],
                    ((-3, -2), (-7, -6)),
                    None,
                    None,
                ],
                [
                    [(-3, -7, 0), (1, 2, 0)],
                    [],
                    ((-3, 1), (-7, 2)),
                    None,
                    None,
                ],
                [
                    [(0, 0, 2)],
                    [],
                    None,
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(2)),
                ],
                [
                    [(-3, -7, 0), (1, 2, 1)],
                    [],
                    None,
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(3)),
                ],
                [
                    [(-3, -7, 0), (1, 2, 1), (3, 3, 2)],
                    [],
                    ((1, 1), (2, 2)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                ],
                [
                    [(0, 0, -2)],
                    [],
                    None,
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                ],
                [
                    [(-3, -7, 0), (1, 2, -1)],
                    [],
                    None,
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(3)),
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                ],
                [
                    [(-3, -7, 0), (1, 2, -1), (3, 3, -2)],
                    [],
                    ((1, 1), (2, 2)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(3)),
                ],
                [
                    [(1, 0, 0)],
                    [(0, 1, 0)],
                    ((0, 1), (0, 1)),
                    None,
                    None,
                ],
                [
                    [(-3, -7, 0), (3, 3, -2)],
                    [
                        (1, 2, -1),
                    ],
                    ((1, 1), (2, 2)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(3)),
                ],
            ],
        )
        def test_expand(
            self,
            read_offsets_1,
            read_offsets_2,
            iteration_offset,
            write_node_interval,
            read_node_interval,
        ):
            if write_node_interval is None:
                write_node_interval = Interval(start=AxisBound.start(), end=AxisBound.end())
            if read_node_interval is None:
                read_node_interval = Interval(start=AxisBound.start(), end=AxisBound.end())

            read_expression = FieldAccess(
                name="tmp1",
                offset=CartesianOffset(**{k: v for k, v in zip("ijk", read_offsets_1[0])}),
            )
            for offset in read_offsets_1[1:]:
                read_expression = BinaryOp(
                    op=ArithmeticOperator.ADD,
                    left=read_expression,
                    right=FieldAccess(
                        name="tmp1", offset=CartesianOffset(**{k: v for k, v in zip("ijk", offset)})
                    ),
                )
            for offset in read_offsets_2:
                read_expression = BinaryOp(
                    op=ArithmeticOperator.ADD,
                    left=read_expression,
                    right=FieldAccess(
                        name="tmp2", offset=CartesianOffset(**{k: v for k, v in zip("ijk", offset)})
                    ),
                )

            write_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp1", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    ),
                    AssignStmt(
                        left=FieldAccess(name="tmp2", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    ),
                ],
                mask=None,
                iteration_space=None,
            )
            read_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="out", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=read_expression,
                    ),
                ],
                mask=None,
                iteration_space=None,
            )

            iteration_offset_dict = {
                read_node.id_: CartesianIterationOffset(i_offsets=(0, 0), j_offsets=(0, 0))
            }
            _dependency_expansion_backward(
                read_node,
                read_node_interval,
                write_node,
                write_node_interval,
                iteration_offset_dict,
            )

            assert write_node.id_ in iteration_offset_dict
            if iteration_offset is None:
                assert iteration_offset_dict[write_node.id_] is None
            else:
                assert iteration_offset_dict[write_node.id_].i_offsets == iteration_offset[0]
                assert iteration_offset_dict[write_node.id_].j_offsets == iteration_offset[1]

        def test_overwrite_none(self):
            write_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                mask=None,
                iteration_space=None,
            )
            read_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="out", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=1, k=0)),
                    )
                ],
                mask=None,
                iteration_space=None,
            )
            iteration_offsets = {
                read_node.id_: CartesianIterationOffset(i_offsets=(0, 0), j_offsets=(0, 0)),
                write_node.id_: None,
            }
            _dependency_expansion_backward(
                read_node,
                Interval(start=AxisBound.start(), end=AxisBound.end()),
                write_node,
                Interval(start=AxisBound.start(), end=AxisBound.end()),
                iteration_offsets,
            )
            assert write_node.id_ in iteration_offsets
            assert iteration_offsets[write_node.id_].i_offsets == (0, 0)
            assert iteration_offsets[write_node.id_].j_offsets == (1, 1)

        def test_not_overwrite_with_none(self):

            write_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                mask=None,
                iteration_space=None,
            )
            write_node_interval = Interval(start=AxisBound.start(), end=AxisBound.end())
            read_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="2.0", dtype=DataType.FLOAT64),
                    )
                ],
                mask=None,
                iteration_space=None,
            )
            read_node_interval = Interval(start=AxisBound.start(), end=AxisBound.end())

            iteration_offset_dict = {
                write_node.id_: CartesianIterationOffset(i_offsets=(1, 1), j_offsets=(1, 1)),
                read_node.id_: CartesianIterationOffset(i_offsets=(0, 0), j_offsets=(0, 0)),
            }
            _dependency_expansion_backward(
                read_node,
                read_node_interval,
                write_node,
                write_node_interval,
                iteration_offset_dict,
            )
            assert write_node.id_ in iteration_offset_dict
            assert iteration_offset_dict[write_node.id_] is not None
            assert iteration_offset_dict[write_node.id_].i_offsets == (1, 1)
            assert iteration_offset_dict[write_node.id_].j_offsets == (1, 1)

        def test_overwrite_no_expand(self):

            write_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                mask=None,
                iteration_space=None,
            )
            write_node_interval = Interval(start=AxisBound.start(), end=AxisBound.end())
            read_node = HorizontalExecution(
                declarations=[],
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="2.0", dtype=DataType.FLOAT64),
                    )
                ],
                mask=None,
                iteration_space=None,
            )
            read_node_interval = Interval(start=AxisBound.start(), end=AxisBound.end())

            iteration_offset_dict = {
                read_node.id_: CartesianIterationOffset(i_offsets=(0, 0), j_offsets=(0, 0))
            }
            _dependency_expansion_backward(
                read_node,
                read_node_interval,
                write_node,
                write_node_interval,
                iteration_offset_dict,
            )
            assert write_node.id_ in iteration_offset_dict
            assert iteration_offset_dict[write_node.id_] is None

        @pytest.mark.parametrize(
            [
                "dependent_iteration_offset",
                "read_offsets",
                "initial_iteration_offset",
                "result_iteration_offset",
            ],
            [
                [((1, 1), (1, 1)), [(1, 1, 0)], ((1, 1), (1, 1)), ((1, 2), (1, 2))],
                [((1, 1), (1, 1)), [(1, 1, 0)], None, ((2, 2), (2, 2))],
                [((0, 0), (0, 1)), [(1, 1, 0)], ((0, 0), (0, 0)), ((0, 1), (0, 2))],
                [((1, 1), (1, 1)), [(1, 0, 0), (0, 1, 0)], ((1, 1), (1, 1)), ((1, 2), (1, 2))],
                [
                    ((1, 1), (1, 1)),
                    [
                        (0, 1, 0),
                        (1, 0, 0),
                    ],
                    None,
                    ((1, 2), (1, 2)),
                ],
                [((0, 0), (0, 1)), [(1, 0, 0), (0, 1, 0)], ((0, 0), (0, 0)), ((0, 1), (0, 2))],
                [
                    ((1, 1), (1, 1)),
                    [
                        (0, 1, 0),
                        (1, 0, 0),
                    ],
                    None,
                    ((1, 2), (1, 2)),
                ],
            ],
        )
        def test_stack(
            self,
            dependent_iteration_offset,
            read_offsets,
            initial_iteration_offset,
            result_iteration_offset,
        ):

            write_node = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp1", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                mask=None,
                iteration_space=None,
                declarations=[],
            )
            write_node_interval = Interval(start=AxisBound.start(), end=AxisBound.end())

            read_expression = FieldAccess(
                name="tmp1",
                offset=CartesianOffset(**{k: v for k, v in zip("ijk", read_offsets[0])}),
            )
            for offset in read_offsets[1:]:
                read_expression = BinaryOp(
                    op=ArithmeticOperator.ADD,
                    left=read_expression,
                    right=FieldAccess(
                        name="tmp1", offset=CartesianOffset(**{k: v for k, v in zip("ijk", offset)})
                    ),
                )

            read_node = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(name="tmp2", offset=CartesianOffset(i=0, j=0, k=0)),
                        right=read_expression,
                    )
                ],
                mask=None,
                iteration_space=None,
                declarations=[],
            )
            read_node_interval = Interval(start=AxisBound.start(), end=AxisBound.end())

            initial_iteration_offset = (
                CartesianIterationOffset(
                    i_offsets=initial_iteration_offset[0], j_offsets=initial_iteration_offset[1]
                )
                if initial_iteration_offset is not None
                else None
            )
            dependent_iteration_offset = (
                CartesianIterationOffset(
                    i_offsets=dependent_iteration_offset[0], j_offsets=dependent_iteration_offset[1]
                )
                if dependent_iteration_offset is not None
                else None
            )
            iteration_offset_dict = {
                write_node.id_: initial_iteration_offset,
                read_node.id_: dependent_iteration_offset,
            }
            _dependency_expansion_backward(
                read_node,
                read_node_interval,
                write_node,
                write_node_interval,
                iteration_offset_dict,
            )
            assert write_node.id_ in iteration_offset_dict
            assert iteration_offset_dict[write_node.id_].i_offsets == result_iteration_offset[0]
            assert iteration_offset_dict[write_node.id_].j_offsets == result_iteration_offset[1]

    class TestComputeIterationOffsets:
        def test_double_outputs(self):
            write1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="out1",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                declarations=[],
            )
            write2 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="out2",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=BinaryOp(
                            op=ArithmeticOperator.ADD,
                            left=FieldAccess(
                                name="out1",
                                offset=CartesianOffset(i=1, j=0, k=0),
                                dtype=DataType.FLOAT64,
                            ),
                            right=FieldAccess(
                                name="out1",
                                offset=CartesianOffset(i=-1, j=0, k=0),
                                dtype=DataType.FLOAT64,
                            ),
                            dtype=DataType.FLOAT64,
                        ),
                    )
                ],
                declarations=[],
            )
            vertical_loop = VerticalLoop(
                interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                horizontal_executions=[write1, write2],
                loop_order=LoopOrder.PARALLEL,
                declarations=[],
            )
            params = [
                FieldDecl(name="out1", dtype=DataType.FLOAT64),
                FieldDecl(name="out2", dtype=DataType.FLOAT64),
            ]
            stencil = Stencil(
                name="test_double_outputs", params=params, vertical_loops=[vertical_loop]
            )
            graph = OIRHorizontalExecutionDependencyGraphBuilder.apply(stencil)
            assert len(graph.edges) == 1
            assert (write1.id_, write2.id_) in graph.edges

            res = _compute_iteration_offsets(stencil, graph)
            assert res[write1.id_].i_offsets == (-1, 1)
            assert res[write1.id_].j_offsets == (0, 0)
            assert res[write2.id_].i_offsets == (0, 0)
            assert res[write2.id_].j_offsets == (0, 0)

        def test_independent_component_none(self):
            write1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="out",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                declarations=[],
            )
            independent1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                declarations=[Temporary(name="tmp", dtype=DataType.FLOAT64)],
            )
            vertical_loop = VerticalLoop(
                interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                horizontal_executions=[write1, independent1],
                loop_order=LoopOrder.PARALLEL,
                declarations=[],
            )
            params = [
                FieldDecl(name="out", dtype=DataType.FLOAT64),
            ]
            stencil = Stencil(
                name="test_independent_component_none",
                params=params,
                vertical_loops=[vertical_loop],
            )
            graph = OIRHorizontalExecutionDependencyGraphBuilder.apply(stencil)
            assert len(graph.edges) == 0

            res = _compute_iteration_offsets(stencil, graph)
            assert res[write1.id_].i_offsets == (0, 0)
            assert res[write1.id_].j_offsets == (0, 0)
            assert res[independent1.id_] is None

        def test_dead_end_none(self):
            write1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="out",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                declarations=[],
            )
            independent1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp1",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=FieldAccess(
                            name="out",
                            offset=CartesianOffset(i=0, j=1, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                    )
                ],
                declarations=[Temporary(name="tmp", dtype=DataType.FLOAT64)],
            )
            independent2 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp2",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=FieldAccess(
                            name="out",
                            offset=CartesianOffset(i=1, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                    )
                ],
                declarations=[Temporary(name="tmp", dtype=DataType.FLOAT64)],
            )
            independent3 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp3",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=BinaryOp(
                            op=ArithmeticOperator.ADD,
                            dtype=DataType.FLOAT64,
                            left=FieldAccess(
                                name="tmp1",
                                offset=CartesianOffset(i=1, j=0, k=0),
                                dtype=DataType.FLOAT64,
                            ),
                            right=FieldAccess(
                                name="tmp2",
                                offset=CartesianOffset(i=0, j=1, k=0),
                                dtype=DataType.FLOAT64,
                            ),
                        ),
                    )
                ],
                declarations=[],
            )
            vertical_loop = VerticalLoop(
                interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                horizontal_executions=[write1, independent1, independent2, independent3],
                loop_order=LoopOrder.PARALLEL,
                declarations=[
                    Temporary(name="tmp1", dtype=DataType.FLOAT64),
                    Temporary(name="tmp2", dtype=DataType.FLOAT64),
                    Temporary(name="tmp3", dtype=DataType.FLOAT64),
                ],
            )
            params = [
                FieldDecl(name="out", dtype=DataType.FLOAT64),
            ]
            stencil = Stencil(
                name="test_dead_end_none",
                params=params,
                vertical_loops=[vertical_loop],
            )
            graph = OIRHorizontalExecutionDependencyGraphBuilder.apply(stencil)
            assert len(graph.edges) == 4
            assert (write1.id_, independent1.id_) in graph.edges
            assert (write1.id_, independent2.id_) in graph.edges
            assert (independent1.id_, independent3.id_) in graph.edges
            assert (independent2.id_, independent3.id_) in graph.edges

            res = _compute_iteration_offsets(stencil, graph)
            assert res[write1.id_].i_offsets == (0, 0)
            assert res[write1.id_].j_offsets == (0, 0)
            assert res[independent1.id_] is None
            assert res[independent2.id_] is None
            assert res[independent3.id_] is None

        def test_intervals_interlock(self):
            tmp1 = VerticalLoop(
                interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                horizontal_executions=[
                    HorizontalExecution(
                        body=[
                            AssignStmt(
                                left=FieldAccess(
                                    name="tmp1",
                                    offset=CartesianOffset(i=0, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                                right=Literal(value="1.0", dtype=DataType.FLOAT64),
                            )
                        ],
                        declarations=[],
                    )
                ],
                loop_order=LoopOrder.PARALLEL,
                declarations=[],
            )
            tmp2 = VerticalLoop(
                interval=Interval(start=AxisBound.start(), end=AxisBound.from_start(3)),
                horizontal_executions=[
                    HorizontalExecution(
                        body=[
                            AssignStmt(
                                left=FieldAccess(
                                    name="tmp2",
                                    offset=CartesianOffset(i=0, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                                right=FieldAccess(
                                    name="tmp1",
                                    offset=CartesianOffset(i=0, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                            )
                        ],
                        declarations=[],
                    )
                ],
                loop_order=LoopOrder.PARALLEL,
                declarations=[],
            )

            tmp3 = VerticalLoop(
                interval=Interval(start=AxisBound.from_end(-3), end=AxisBound.end()),
                horizontal_executions=[
                    HorizontalExecution(
                        body=[
                            AssignStmt(
                                left=FieldAccess(
                                    name="tmp2",
                                    offset=CartesianOffset(i=0, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                                right=FieldAccess(
                                    name="tmp1",
                                    offset=CartesianOffset(i=1, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                            )
                        ],
                        declarations=[],
                    )
                ],
                loop_order=LoopOrder.PARALLEL,
                declarations=[],
            )

            write1 = VerticalLoop(
                interval=Interval(start=AxisBound.start(), end=AxisBound.from_start(3)),
                horizontal_executions=[
                    HorizontalExecution(
                        body=[
                            AssignStmt(
                                left=FieldAccess(
                                    name="out",
                                    offset=CartesianOffset(i=0, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                                right=FieldAccess(
                                    name="tmp2",
                                    offset=CartesianOffset(i=2, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                            )
                        ],
                        declarations=[],
                    )
                ],
                loop_order=LoopOrder.PARALLEL,
                declarations=[],
            )
            write2 = VerticalLoop(
                interval=Interval(start=AxisBound.from_start(3), end=AxisBound.from_end(-3)),
                horizontal_executions=[
                    HorizontalExecution(
                        body=[
                            AssignStmt(
                                left=FieldAccess(
                                    name="out",
                                    offset=CartesianOffset(i=0, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                                right=FieldAccess(
                                    name="tmp1",
                                    offset=CartesianOffset(i=0, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                            )
                        ],
                        declarations=[],
                    )
                ],
                loop_order=LoopOrder.PARALLEL,
                declarations=[],
            )
            write3 = VerticalLoop(
                interval=Interval(start=AxisBound.from_end(-3), end=AxisBound.end()),
                horizontal_executions=[
                    HorizontalExecution(
                        body=[
                            AssignStmt(
                                left=FieldAccess(
                                    name="out",
                                    offset=CartesianOffset(i=0, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                                right=FieldAccess(
                                    name="tmp2",
                                    offset=CartesianOffset(i=1, j=0, k=0),
                                    dtype=DataType.FLOAT64,
                                ),
                            )
                        ],
                        declarations=[],
                    )
                ],
                loop_order=LoopOrder.PARALLEL,
                declarations=[],
            )

            params = [
                FieldDecl(name="out", dtype=DataType.FLOAT64),
                Temporary(name="tmp1", dtype=DataType.FLOAT64),
                Temporary(name="tmp2", dtype=DataType.FLOAT64),
            ]
            stencil = Stencil(
                name="test_dag_stack",
                params=params,
                vertical_loops=[tmp1, tmp2, tmp3, write1, write2, write3],
            )
            graph = OIRHorizontalExecutionDependencyGraphBuilder.apply(stencil)
            assert len(graph.edges) == 5
            assert (
                tmp1.horizontal_executions[0].id_,
                tmp2.horizontal_executions[0].id_,
            ) in graph.edges
            assert (
                tmp1.horizontal_executions[0].id_,
                tmp3.horizontal_executions[0].id_,
            ) in graph.edges
            assert (
                tmp2.horizontal_executions[0].id_,
                write1.horizontal_executions[0].id_,
            ) in graph.edges
            assert (
                tmp1.horizontal_executions[0].id_,
                write2.horizontal_executions[0].id_,
            ) in graph.edges
            assert (
                tmp3.horizontal_executions[0].id_,
                write3.horizontal_executions[0].id_,
            ) in graph.edges

            res = _compute_iteration_offsets(stencil, graph)

            assert res[write1.horizontal_executions[0].id_].i_offsets == (0, 0)
            assert res[write1.horizontal_executions[0].id_].j_offsets == (0, 0)

            assert res[write2.horizontal_executions[0].id_].i_offsets == (0, 0)
            assert res[write2.horizontal_executions[0].id_].j_offsets == (0, 0)

            assert res[write3.horizontal_executions[0].id_].i_offsets == (0, 0)
            assert res[write3.horizontal_executions[0].id_].j_offsets == (0, 0)

            assert res[tmp3.horizontal_executions[0].id_].i_offsets == (0, 1)
            assert res[tmp3.horizontal_executions[0].id_].j_offsets == (0, 0)

            assert res[tmp2.horizontal_executions[0].id_].i_offsets == (0, 2)
            assert res[tmp2.horizontal_executions[0].id_].j_offsets == (0, 0)

            assert res[tmp1.horizontal_executions[0].id_].i_offsets == (0, 2)
            assert res[tmp1.horizontal_executions[0].id_].j_offsets == (0, 0)

        def test_masks_are_read(self):
            write1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.BOOL,
                        ),
                        right=Literal(value="true", dtype=DataType.BOOL),
                    )
                ],
                declarations=[],
            )
            write2 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="out",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                mask=FieldAccess(
                    name="tmp",
                    offset=CartesianOffset(i=0, j=1, k=0),
                    dtype=DataType.BOOL,
                ),
                declarations=[],
            )
            vertical_loop = VerticalLoop(
                interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                horizontal_executions=[write1, write2],
                loop_order=LoopOrder.PARALLEL,
                declarations=[FieldDecl(name="tmp", dtype=DataType.BOOL)],
            )
            params = [
                FieldDecl(name="out", dtype=DataType.FLOAT64),
            ]
            stencil = Stencil(
                name="test_masks_are_read", params=params, vertical_loops=[vertical_loop]
            )
            graph = OIRHorizontalExecutionDependencyGraphBuilder.apply(stencil)
            assert len(graph.edges) == 1
            assert (write1.id_, write2.id_) in graph.edges

            res = _compute_iteration_offsets(stencil, graph)
            assert res[write1.id_].i_offsets == (0, 0)
            assert res[write1.id_].j_offsets == (1, 1)
            assert res[write2.id_].i_offsets == (0, 0)
            assert res[write2.id_].j_offsets == (0, 0)

        def test_dag_stack(self):
            tmp1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp1",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                declarations=[],
            )
            tmp2 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp2",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=FieldAccess(
                            name="tmp1",
                            offset=CartesianOffset(i=0, j=1, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                    )
                ],
                declarations=[],
            )
            tmp3 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp3",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=FieldAccess(
                            name="tmp1",
                            offset=CartesianOffset(i=1, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                    )
                ],
                declarations=[],
            )
            write1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="out",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=BinaryOp(
                            op=ArithmeticOperator.ADD,
                            dtype=DataType.FLOAT64,
                            left=FieldAccess(
                                name="tmp2",
                                offset=CartesianOffset(i=1, j=0, k=0),
                                dtype=DataType.FLOAT64,
                            ),
                            right=FieldAccess(
                                name="tmp3",
                                offset=CartesianOffset(i=0, j=1, k=0),
                                dtype=DataType.FLOAT64,
                            ),
                        ),
                    )
                ],
                declarations=[],
            )
            vertical_loop = VerticalLoop(
                interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                horizontal_executions=[tmp1, tmp2, tmp3, write1],
                loop_order=LoopOrder.PARALLEL,
                declarations=[
                    Temporary(name="tmp1", dtype=DataType.FLOAT64),
                    Temporary(name="tmp2", dtype=DataType.FLOAT64),
                    Temporary(name="tmp3", dtype=DataType.FLOAT64),
                ],
            )
            params = [
                FieldDecl(name="out", dtype=DataType.FLOAT64),
            ]
            stencil = Stencil(
                name="test_dag_stack",
                params=params,
                vertical_loops=[vertical_loop],
            )
            graph = OIRHorizontalExecutionDependencyGraphBuilder.apply(stencil)
            assert len(graph.edges) == 4
            assert (tmp1.id_, tmp2.id_) in graph.edges
            assert (tmp1.id_, tmp3.id_) in graph.edges
            assert (tmp2.id_, write1.id_) in graph.edges
            assert (tmp3.id_, write1.id_) in graph.edges

            res = _compute_iteration_offsets(stencil, graph)

            assert res[write1.id_].i_offsets == (0, 0)
            assert res[write1.id_].j_offsets == (0, 0)

            assert res[tmp1.id_].i_offsets == (1, 1)
            assert res[tmp1.id_].j_offsets == (1, 1)

            assert res[tmp2.id_].i_offsets == (1, 1)
            assert res[tmp2.id_].j_offsets == (0, 0)

            assert res[tmp3.id_].i_offsets == (0, 0)
            assert res[tmp3.id_].j_offsets == (1, 1)

        def test_shadowed_out_is_none(self):

            write1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="out",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                declarations=[],
            )
            write2 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="out",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=Literal(value="2.0", dtype=DataType.FLOAT64),
                    )
                ],
                declarations=[],
            )
            vertical_loop = VerticalLoop(
                interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                horizontal_executions=[write1, write2],
                loop_order=LoopOrder.PARALLEL,
                declarations=[],
            )
            params = [
                FieldDecl(name="out", dtype=DataType.FLOAT64),
            ]
            stencil = Stencil(
                name="test_none_propagates",
                params=params,
                vertical_loops=[vertical_loop],
            )
            graph = OIRHorizontalExecutionDependencyGraphBuilder.apply(stencil)
            assert len(graph.edges) == 1
            assert (write1.id_, write2.id_) in graph.edges

            res = _compute_iteration_offsets(stencil, graph)

            assert res[write2.id_].i_offsets == (0, 0)
            assert res[write2.id_].j_offsets == (0, 0)
            assert res[write1.id_] is None

        def test_none_propagates(self):
            tmp1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp1",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=Literal(value="1.0", dtype=DataType.FLOAT64),
                    )
                ],
                declarations=[],
            )
            tmp2 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp2",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=FieldAccess(
                            name="tmp1",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                    )
                ],
                declarations=[],
            )
            tmp3 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="tmp3",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=FieldAccess(
                            name="tmp1",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                    )
                ],
                declarations=[],
            )
            write1 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="out",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=BinaryOp(
                            op=ArithmeticOperator.ADD,
                            dtype=DataType.FLOAT64,
                            left=FieldAccess(
                                name="tmp2",
                                offset=CartesianOffset(i=0, j=0, k=0),
                                dtype=DataType.FLOAT64,
                            ),
                            right=FieldAccess(
                                name="tmp3",
                                offset=CartesianOffset(i=0, j=0, k=0),
                                dtype=DataType.FLOAT64,
                            ),
                        ),
                    )
                ],
                declarations=[],
            )
            write2 = HorizontalExecution(
                body=[
                    AssignStmt(
                        left=FieldAccess(
                            name="out",
                            offset=CartesianOffset(i=0, j=0, k=0),
                            dtype=DataType.FLOAT64,
                        ),
                        right=Literal(value="2.0", dtype=DataType.FLOAT64),
                    )
                ],
                declarations=[],
            )
            vertical_loop = VerticalLoop(
                interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                horizontal_executions=[tmp1, tmp2, tmp3, write1, write2],
                loop_order=LoopOrder.PARALLEL,
                declarations=[
                    Temporary(name="tmp1", dtype=DataType.FLOAT64),
                    Temporary(name="tmp2", dtype=DataType.FLOAT64),
                    Temporary(name="tmp3", dtype=DataType.FLOAT64),
                ],
            )
            params = [
                FieldDecl(name="out", dtype=DataType.FLOAT64),
            ]
            stencil = Stencil(
                name="test_none_propagates",
                params=params,
                vertical_loops=[vertical_loop],
            )
            graph = OIRHorizontalExecutionDependencyGraphBuilder.apply(stencil)
            assert len(graph.edges) == 5
            assert (tmp1.id_, tmp2.id_) in graph.edges
            assert (tmp1.id_, tmp3.id_) in graph.edges
            assert (tmp2.id_, write1.id_) in graph.edges
            assert (tmp3.id_, write1.id_) in graph.edges
            assert (write1.id_, write2.id_) in graph.edges

            res = _compute_iteration_offsets(stencil, graph)

            assert res[write2.id_].i_offsets == (0, 0)
            assert res[write2.id_].j_offsets == (0, 0)
            assert res[write1.id_] is None

            assert res[tmp1.id_] is None
            assert res[tmp2.id_] is None
            assert res[tmp3.id_] is None
