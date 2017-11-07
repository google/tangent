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
    numpy.shape, numpy.zeros, numpy.ones, numpy.zeros_like, numpy.ones_like,
])


def register_non_differentiable_functions(*funcs):
  global NON_DIFFERENTIABLE
  NON_DIFFERENTIABLE |= set(funcs)
