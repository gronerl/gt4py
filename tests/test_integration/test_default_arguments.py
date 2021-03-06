# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
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

import numpy as np

import gt4py as gt
import gt4py.gtscript as gtscript
import gt4py.backend as gt_backend
import gt4py.storage as gt_storage
from gt4py.gtscript import Field
import pytest


def a_stencil(
    arg1: Field[np.float64],
    arg2: Field[np.float64],
    arg3: Field[np.float64] = None,
    *,
    par1: np.float64,
    par2: np.float64 = 7.0,
    par3: np.float64 = None,
):
    from __externals__ import BRANCH

    with computation(PARALLEL), interval(...):

        if BRANCH:
            arg1 = arg1 * par1 * par2
        else:
            arg1 = arg2 + arg3 * par1 * par2 * par3


@pytest.mark.parametrize(
    "backend",
    [
        name
        for name in gt_backend.REGISTRY.names
        if gt_backend.from_name(name).storage_info["device"] == "cpu"
    ],
)
def test_default_arguments(backend):
    branch_true = gtscript.stencil(
        backend=backend, definition=a_stencil, externals={"BRANCH": True}, rebuild=True
    )
    branch_false = gtscript.stencil(
        backend=backend, definition=a_stencil, externals={"BRANCH": False}, rebuild=True
    )

    arg1 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    arg2 = gt_storage.zeros(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    arg3 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    tmp = np.asarray(arg3)
    tmp *= 2

    branch_true(arg1, None, arg3, par1=2.0)
    np.testing.assert_equal(arg1, 14 * np.ones((3, 3, 3)))
    branch_true(arg1, None, par1=2.0)
    np.testing.assert_equal(arg1, 196 * np.ones((3, 3, 3)))
    branch_false(arg1, arg2, arg3, par1=2.0, par3=2.0)
    np.testing.assert_equal(arg1, 56 * np.ones((3, 3, 3)))
    try:
        branch_false(arg1, arg2, par1=2.0, par3=2.0)
    except ValueError:
        pass
    else:
        assert False

    arg1 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    arg2 = gt_storage.zeros(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    arg3 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    tmp = np.asarray(arg3)
    tmp *= 2

    branch_true(arg1, arg2=None, par1=2.0, par2=5, par3=3.0)
    np.testing.assert_equal(arg1, 10 * np.ones((3, 3, 3)))
    branch_true(arg1, arg2=None, par1=2.0, par2=5)
    np.testing.assert_equal(arg1, 100 * np.ones((3, 3, 3)))
    branch_false(arg1, arg2, arg3, par1=2.0, par2=5, par3=3.0)
    np.testing.assert_equal(arg1, 60 * np.ones((3, 3, 3)))
    try:
        branch_false(arg1, arg2, arg3, par1=2.0, par2=5)
    except ValueError:
        pass
    else:
        assert False
