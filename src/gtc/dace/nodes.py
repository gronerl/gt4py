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

from typing import Dict

import dace.data
import dace.dtypes
import dace.properties
import dace.subsets
from dace import library

from gt4py.definitions import Extent
from gtc.oir import VerticalLoop


@library.node
class StencilComputation(library.LibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {}
    default_implementation = "default"

    oir_node = dace.properties.DataclassProperty(dtype=VerticalLoop, default=None, allow_none=True)
    extents = dace.properties.DictProperty(
        key_type=int, value_type=Extent, default=None, allow_none=True
    )

    _dace_library_name = "StencilComputation"

    def __init__(
        self,
        name="unnamed_vloop",
        oir_node: VerticalLoop = None,
        extents: Dict[int, Extent] = None,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)

        if oir_node is not None:
            self.oir_node = oir_node
            self.extents = extents
            if oir_node.loc is not None:

                self.debuginfo = dace.dtypes.DebugInfo(
                    oir_node.loc.line,
                    oir_node.loc.column,
                    oir_node.loc.line,
                    oir_node.loc.column,
                    oir_node.loc.source,
                )
