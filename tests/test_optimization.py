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
import gast
import pytest

from tangent import optimization
from tangent import quoting


def test_assignment_propagation():
  def f(x):
    y = x
    z = y
    return z

  node = quoting.parse_function(f)
  node = optimization.assignment_propagation(node)
  assert len(node.body[0].body) == 2


def test_dce():
  def f(x):
    y = 2 * x
    return x

  node = quoting.parse_function(f)
  node = optimization.dead_code_elimination(node)
  assert isinstance(node.body[0].body[0], gast.Return)


def test_fixed_point():
  def f(x):
    y = g(x)
    z = h(y)
    return x

  node = quoting.parse_function(f)
  node = optimization.optimize(node)
  assert isinstance(node.body[0].body[0], gast.Return)


def test_constant_folding():
  def f(x):
    x = 1 * x
    x = 0 * x
    x = x * 1
    x = x * 0
    x = x * 2
    x = 2 * x
    x = 2 * 3
    x = 1 + x
    x = 0 + x
    x = x + 1
    x = x + 0
    x = x + 2
    x = 2 + x
    x = 2 + 3
    x = 1 - x
    x = 0 - x
    x = x - 1
    x = x - 0
    x = x - 2
    x = 2 - x
    x = 2 - 3
    x = 1 / x
    x = 0 / x
    x = x / 1
    x = x / 0
    x = x / 2
    x = 2 / x
    x = 2 / 8
    x = 1 ** x
    x = 0 ** x
    x = x ** 1
    x = x ** 0
    x = x ** 2
    x = 2 ** x
    x = 2 ** 3

  def f_opt(x):
    x = x
    x = 0
    x = x
    x = 0
    x = x * 2
    x = 2 * x
    x = 6
    x = 1 + x
    x = x
    x = x + 1
    x = x
    x = x + 2
    x = 2 + x
    x = 5
    x = 1 - x
    x = -x
    x = x - 1
    x = x
    x = x - 2
    x = 2 - x
    x = -1
    x = 1 / x
    x = 0 / x
    x = x
    x = x / 0
    x = x / 2
    x = 2 / x
    x = 0.25
    x = 1
    x = 0
    x = x
    x = 1
    x = x ** 2
    x = 2 ** x
    x = 8

  node = quoting.parse_function(f)
  node = optimization.constant_folding(node)
  node_opt = quoting.parse_function(f_opt)
  lines = quoting.to_source(node).strip().split('\n')[1:]
  lines_opt = quoting.to_source(node_opt).strip().split('\n')[1:]
  # In Python 2 integer division could be on, in which case...
  if 1 / 2 == 0:
    lines_opt[27] = '    x = 0'
  assert lines == lines_opt


if __name__ == '__main__':
  assert not pytest.main([__file__])
