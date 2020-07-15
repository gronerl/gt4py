# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
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

import itertools

import numpy as np
import pytest

import gt4py as gt
from gt4py import gtscript
from gt4py import backend as gt_backend
from gt4py import storage as gt_storage

from .stencil_definitions import REGISTRY as stencil_definitions
from .stencil_definitions import EXTERNALS_REGISTRY as externals_registry
from ..definitions import ALL_BACKENDS, CPU_BACKENDS, GPU_BACKENDS, INTERNAL_BACKENDS


@pytest.mark.parametrize(
    ["name", "backend"], itertools.product(stencil_definitions.names, CPU_BACKENDS)
)
def test_generation_cpu(name, backend):
    stencil_definition = stencil_definitions[name]
    externals = externals_registry[name]
    stencil = gtscript.stencil(backend, stencil_definition, externals=externals)
    args = {}
    for k, v in stencil_definition.__annotations__.items():
        if isinstance(v, gtscript._FieldDescriptor):
            args[k] = gt_storage.ones(
                dtype=v.dtype,
                mask=gtscript.mask_from_axes(v.axes),
                backend=backend,
                shape=(23, 23, 23),
                default_origin=(10, 10, 10),
            )
        else:
            args[k] = v(1.5)
    stencil(**args, origin=(10, 10, 10), domain=(3, 3, 3))


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ["name", "backend"], itertools.product(stencil_definitions.names, GPU_BACKENDS)
)
def test_generation_gpu(name, backend):
    stencil_definition = stencil_definitions[name]
    externals = externals_registry[name]
    stencil = gtscript.stencil(backend, stencil_definition, externals=externals)
    args = {}
    for k, v in stencil_definition.__annotations__.items():
        if isinstance(v, gtscript._FieldDescriptor):
            args[k] = gt_storage.ones(
                dtype=v.dtype,
                mask=gtscript.mask_from_axes(v.axes),
                backend=backend,
                shape=(23, 23, 23),
                default_origin=(10, 10, 10),
            )
        else:
            args[k] = v(1.5)
    stencil(**args, origin=(10, 10, 10), domain=(3, 3, 3))


class TestTemporaryDeclarationsInConditionals:
    def test_defined_both_same(self):
        """It is fine to write to field_b the first time in a conditional, if the implicit type
        is the same in both the if.. and else.. branch"""

        @gtscript.stencil(backend="debug")
        def definition(field_a: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                if field_a < 0:
                    field_b = -field_a
                else:
                    field_b = field_a
                field_a = field_b

    def test_defined_both_different(self):
        """field_b is used to write a floating point value once and once with integer, this is
        ambiguous."""

        from gt4py.frontend.gtscript_frontend import GTScriptDataTypeError

        with pytest.raises(
            GTScriptDataTypeError, match="Symbol 'field_b' used with inconsistent data types."
        ):

            @gtscript.stencil(backend="debug")
            def definition(field_a: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    if field_a < 0:
                        field_b = 0.0
                    else:
                        field_b = 0
                    field_a = field_b

    def test_defined_single_read_later(self):
        """field_b is declared in a conditional such that potentially undefined values remain,
        this is disallowed"""

        from gt4py.frontend.gtscript_frontend import GTScriptDefinitionError

        with pytest.raises(GTScriptDefinitionError):

            @gtscript.stencil(backend="debug")
            def definition(field_a: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    if field_a < 0:
                        field_b = 0.0
                    field_a = field_b

        with pytest.raises(
            GTScriptDefinitionError,
            match="Temporary 'field_b' declared in only one branch of conditional but accessed "
            "again outside of conditional",
        ):

            @gtscript.stencil(backend="debug")
            def definition(field_a: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    if field_a < 0:
                        field_a = 0.0
                    else:
                        field_b = 0.0
                    field_a = field_b

    def test_defined_single_local(self):
        """field_b is declared in only one branch of the conditional but is only used there, this
        is fine."""

        @gtscript.stencil(backend="debug")
        def definition(field_a: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                if field_a < 0:
                    field_b = 0.0
                    field_a = field_b
                field_a = field_a + 1

        @gtscript.stencil(backend="debug")
        def definition(field_a: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                if field_a < 0:
                    field_b = 0.0
                    if field_a:
                        field_b = 1.0
                    else:
                        field_b = 2
                field_a = field_a + 1


@gtscript.function
def plus_one(in_field):
    tmp = in_field + 1
    return tmp


def test_subroutine_in_conditional():
    @gtscript.stencil(backend="debug")
    def definition(field_a: gtscript.Field[np.float_], field_b: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            field_c = 0
            if field_a < 0:
                field_b = plus_one(field_a)
