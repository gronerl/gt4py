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

"""Python bindings for Gridtools."""

__copyright__ = "ETH Zurich"
__license__ = "gpl3"


from . import config
from . import utils

from . import definitions
from . import gtscript

from . import ir
from . import analysis
from . import frontend
from . import backend
from . import stencil_object
from . import loader
from . import storage
from . import testing

from .definitions import (
    AccessKind,
    Boundary,
    DomainInfo,
    FieldInfo,
    Grid,
    ParameterInfo,
    CartesianSpace,
)
from .stencil_object import StencilObject

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


import os as _os
import subprocess as _subprocess


def install_gt_sources():
    GIT_BRANCH = "release_v1.1"
    GIT_REPO = "https://github.com/GridTools/gridtools.git"

    install_path = _os.path.dirname(__file__)
    target_path = _os.path.abspath(_os.path.join(install_path, "_external_src", "gridtools"))

    if not _os.path.exists(target_path):
        git_cmd = f"git clone --depth 1 -b {GIT_BRANCH} {GIT_REPO} {target_path}"
        print(f"Getting GridTools C++ sources...\n$ {git_cmd}")
        _subprocess.check_call(git_cmd.split(), stderr=_subprocess.STDOUT)


def remove_gt_sources():
    install_path = _os.path.dirname(__file__)
    target_path = _os.path.abspath(_os.path.join(install_path, "_external_src", "gridtools"))

    if _os.path.exists(target_path):
        rm_cmd = f"rm -Rf {target_path}"
        print(f"Deleting sources...\n$ {rm_cmd}")
        _subprocess.run(rm_cmd.split())


try:
    # Check if GridTools sources are installed
    install_gt_sources()
except Exception:
    print(
        f"Missing GridTools sources. Install them manually in '{_os.path.dirname(__file__)}/_external_src/'"
    )
finally:
    del _os, _subprocess
