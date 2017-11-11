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
"""Forward-mode tests.

Notes
-----
Arguments func, a, b, c, x, and n are automatically filled in.

Pass --short for a quick run.

"""
import sys

import pytest
import tfe_utils
import utils


def test_deriv_unary(func, preserve_result, a):
  """Test derivatives of single-argument scalar functions."""
  utils.test_forward_array(func, (0,), preserve_result, a)


def test_deriv_binary(func, preserve_result, a, b):
  """Test derivatives of two-argument scalar functions."""
  utils.test_forward_array(func, (0,), preserve_result, a, b)


def test_deriv_ternary(func, preserve_result, a, b, c):
  """Test derivatives of three-argument scalar functions."""
  utils.test_forward_array(func, (0,), preserve_result, a, b, c)


def test_deriv_binary_int(func, preserve_result, a, n):
  """Test derivatives of functions with scalar and integer input."""
  utils.test_forward_array(func, (0,), preserve_result, a, n)


def test_deriv_unary_tensor(func, t):
  """Test derivatives of functions with single tensor input."""
  # TODO: remove trace test exemption when tests are consolidated.
  if 'trace' in func.__name__:
    return
  if any(n in func.__name__ for n in ('tfe_rsqrt',)):
    utils.assert_forward_not_implemented(func, (0,))
    return
  tfe_utils.test_forward_tensor(func, (0,), t)


def test_deriv_binary_tensor(func, t1, t2):
  """Test derivatives of functions with binary tensor inputs."""
  if any(n in func.__name__ for n in ('tfe_squared_difference',)):
    utils.assert_forward_not_implemented(func, (0,))
    return
  tfe_utils.test_forward_tensor(func, (0,), t1, t2)
  tfe_utils.test_forward_tensor(func, (1,), t1, t2)


def test_deriv_image(func, timage, tkernel, conv2dstrides):
  """Test derivatives of image functions."""
  utils.assert_forward_not_implemented(func, (0,))


if __name__ == '__main__':
  assert not pytest.main([__file__, '--short'] + sys.argv[1:])
