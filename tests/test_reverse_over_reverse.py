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
"""Tests for reverse-over-reverse automatic differentiation.

Notes
-----
Arguments func, a, b, c, x, and n are automatically filled in.

Pass --short for a quick run.

"""
from autograd import grad as ag_grad
import autograd.numpy as ag_np
import numpy as np
import pytest
import tangent
import utils


def _test_gradgrad_array(func, optimized, *args):
  """Test gradients of functions with NumPy-compatible signatures."""

  def tangent_func():
    func.__globals__['np'] = np
    df = tangent.grad(func, optimized=optimized, verbose=True)
    ddf = tangent.grad(df, optimized=optimized, verbose=True)
    return ddf(*args)

  def reference_func():
    func.__globals__['np'] = ag_np
    return ag_grad(ag_grad(func))(*args)

  def backup_reference_func():
    return utils.numeric_grad(utils.numeric_grad(func))(*args)

  utils.assert_result_matches_reference(
      tangent_func, reference_func, backup_reference_func,
      tolerance=1e-2)  # extra loose bounds for 2nd order grad


def test_reverse_over_reverse_unary(func, a, optimized):
  _test_gradgrad_array(func, optimized, a)


def test_reverse_over_reverse_binary(func, a, b, optimized):
  _test_gradgrad_array(func, optimized, a, b)


def test_reverse_over_reverse_ternary(func, optimized, a, b, c):
  _test_gradgrad_array(func, optimized, a, b, c)


if __name__ == '__main__':
  assert not pytest.main([__file__, '--short'])
