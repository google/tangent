# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
"""Non-differentiable functions.

Not in the mathematical sense, but in the sense of them providing zero gradient
because they provide meta-information (shape) do integer arithmetic, or are
tensor constructors.

Note that one still needs to provide tangents for non-differentiable functions,
but these should simply call the original.
TODO: Remove this requirement.
"""
from __future__ import absolute_import

import numpy
import tangent


NON_DIFFERENTIABLE = set([
    len,
    numpy.alen,
    numpy.eye,
    numpy.identity,
    numpy.shape,
    numpy.zeros,
    numpy.ones,
    numpy.tri,
    numpy.zeros_like,
    numpy.ones_like,
    numpy.floor,
    numpy.ceil,
    numpy.bitwise_and,
    numpy.bitwise_xor,
    numpy.invert,
    numpy.round,
    numpy.rint,
    numpy.around,
    numpy.fix,
    numpy.trunc,
    numpy.all,
    numpy.any,
    numpy.alltrue,
    numpy.sometrue,
    numpy.argmax,
    numpy.argmin,
    numpy.nanargmin,
    numpy.nanargmax,
    numpy.argpartition,
    numpy.argsort,
    numpy.argwhere,
    numpy.nonzero,
    numpy.flatnonzero,
    numpy.count_nonzero,
    numpy.searchsorted,
    numpy.sign,
    numpy.signbit,
    numpy.ndim,
    numpy.rank,
    numpy.floor_divide,
    numpy.logical_and,
    numpy.logical_or,
    numpy.logical_not,
    numpy.logical_xor,
    numpy.isfinite,
    numpy.isinf,
    numpy.isnan,
    numpy.isneginf,
    numpy.isposinf,
    numpy.allclose,
    numpy.isclose,
    numpy.array_equal,
    numpy.array_equiv,
    numpy.greater,
    numpy.greater_equal,
    numpy.less,
    numpy.less_equal,
    numpy.equal,
    numpy.not_equal,
    numpy.iscomplexobj,
    numpy.iscomplex,
    numpy.size,
    numpy.in1d,
    numpy.isscalar,
    numpy.isreal,
    numpy.isrealobj,
    numpy.result_type,
    numpy.arange,
    numpy.empty,
    numpy.empty_like,
    numpy.digitize,
    numpy.bincount,
    numpy.flatnonzero,
    numpy.diag_indices,
    numpy.diag_indices_from,
])


def register_non_differentiable_functions(*funcs):
  global NON_DIFFERENTIABLE
  NON_DIFFERENTIABLE |= set(funcs)
