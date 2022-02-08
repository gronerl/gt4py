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
from gtc.oir import Decl, FieldDecl, VerticalLoop


def set_expansion_order(self, expansion_order):
    if not set(StencilComputation.expansion_order.default) == set(expansion_order) or not len(
        StencilComputation.expansion_order.default
    ) == len(expansion_order):
        raise ValueError(
            "StencilComputation.expansion_order must be a permutation of"
            + str(StencilComputation.expansion_order.default)
        )
    if expansion_order.index("K") < expansion_order.index("Sections"):
        raise ValueError("K iteration must be per section.")
    if any(expansion_order.index(ij) < expansion_order.index("Stages") for ij in "IJ"):
        raise ValueError("I and J iteration must be per stage.")
    if expansion_order.index("Stages") < expansion_order.index("Sections"):
        raise ValueError("Stages are per-section")
    self._expansion_order = list(str(eo) for eo in expansion_order)


@library.node
class StencilComputation(library.LibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {}
    default_implementation = "default"

    oir_node = dace.properties.DataclassProperty(dtype=VerticalLoop, default=None, allow_none=True)
    declarations = dace.properties.DictProperty(
        key_type=str, value_type=Decl, default=None, allow_none=True
    )
    extents = dace.properties.DictProperty(
        key_type=int, value_type=Extent, default=None, allow_none=True
    )
    expansion_order = dace.properties.ListProperty(
        desc="""
        currently supported:
         ["Sections", "Stages", "I", "J", "K"]
         ["Sections", "Stages", "I", "K", "J"]
         ["Sections", "Stages", "J", "I", "K"]
         ["Sections", "Stages", "J", "K", "I"]
         ["Sections", "Stages", "K", "I", "J"]
         ["Sections", "Stages", "K", "J", "I"]
         ["Sections", "K", "Stages", "I", "J"]
         ["Sections", "K", "Stages", "J", "I"]
        """,
        element_type=str,
        default=["Sections", "Stages", "I", "J", "K"],
        allow_none=False,
        setter=set_expansion_order,
    )

    _dace_library_name = "StencilComputation"

    def __init__(
        self,
        name="unnamed_vloop",
        oir_node: VerticalLoop = None,
        extents: Dict[int, Extent] = None,
        declarations: Dict[str, Decl] = None,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)

        if oir_node is not None:
            self.oir_node = oir_node
            self.extents = extents
            self.declarations = declarations
            if oir_node.loc is not None:

                self.debuginfo = dace.dtypes.DebugInfo(
                    oir_node.loc.line,
                    oir_node.loc.column,
                    oir_node.loc.line,
                    oir_node.loc.column,
                    oir_node.loc.source,
                )

    @property
    def field_decls(self) -> Dict[str, FieldDecl]:
        return {
            name: decl for name, decl in self.declarations.items() if isinstance(decl, FieldDecl)
        }
