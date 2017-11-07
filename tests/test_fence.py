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
"""Fence tests."""
import inspect
import sys

import pytest
from tangent import fence
from tangent import quoting

# TODO: Add version-specific checks.
# Not tested (mainly because Python 3.6 is not yet supported):
#   matmult operator
#   f-strings
#   type annotations
#   try/finally (version-specific)
#   try/except (version-specific)
#   tyeld/from
#   await
#   asyncfor
#   asyncwith
#   nonlocal
#   async def

# Note: currently, these tests only cover the rejection cases. Positive cases
# should normally be caught by the main tests.

testglobal = 0


def _assert_tangent_parse_error(func, fragment):
  try:
    fence.validate(quoting.parse_function(func), inspect.getsource(func))
    assert False
  except fence.TangentParseError as expected:
    assert fragment in str(expected)


def test_bytes():
  if sys.version_info >= (3, 0):
    def f(_):
      return b'foo'
    _assert_tangent_parse_error(f, 'Byte Literals')


def test_set():

  def f(_):
    return set({1})

  _assert_tangent_parse_error(f, 'Sets')


def test_del():

  def f(x):
    del x
    return 1

  _assert_tangent_parse_error(f, 'Del')


def test_starred():

  def f(x):
    return zip(*x)

  _assert_tangent_parse_error(f, 'Unpack')


def test_uadd():

  def f(x):
    return +x

  _assert_tangent_parse_error(f, 'Unary Add')


def test_not():

  def f(x):
    return not x

  _assert_tangent_parse_error(f, 'Not operator')


def test_invert():

  def f(x):
    return ~x

  _assert_tangent_parse_error(f, 'Invert')


def test_floordiv():

  def f(x):
    return x // 2

  _assert_tangent_parse_error(f, 'Floor Div')


def test_lshift():

  def f(x):
    return x << 1

  _assert_tangent_parse_error(f, 'Left Shift')


def test_rshift():

  def f(x):
    return x >> 1

  _assert_tangent_parse_error(f, 'Right Shift')


def test_bitor():

  def f(x):
    return x | 1

  _assert_tangent_parse_error(f, 'Bitwise Or')


def test_bitxor():

  def f(x):
    return x ^ 1

  _assert_tangent_parse_error(f, 'Bitwise Xor')


def test_bitand():

  def f(x):
    return x & 1

  _assert_tangent_parse_error(f, 'Bitwise And')


def test_in():

  def f(x):
    return 1 in x

  _assert_tangent_parse_error(f, 'In operator')


def test_notin():

  def f(x):
    return 1 not in x

  _assert_tangent_parse_error(f, 'Not In operator')


def test_ifexp():

  def f(x):
    return 1 if x else 2

  _assert_tangent_parse_error(f, 'Conditional')


def test_setcomp():

  def f(x):
    return {i for i in x}

  _assert_tangent_parse_error(f, 'Set Comprehensions')


def test_generatorexp():

  def f(x):
    return (i for i in x)

  _assert_tangent_parse_error(f, 'Generator')


def test_dictcomp():

  def f(x):
    return {i: 1 for i in x}

  _assert_tangent_parse_error(f, 'Dictionary Comprehensions')


def test_delete():

  def f(x):
    del x[1]
    return x

  _assert_tangent_parse_error(f, 'Delete statements')


def test_import():

  def f(x):
    import tangent
    return x

  _assert_tangent_parse_error(f, 'Import statements')


def test_importfrom():

  def f(x):
    from tangent import grad
    return x

  _assert_tangent_parse_error(f, 'Import/From statements')


def test_alias():

  def f(x):
    import tangent as tg
    return x

  # The checker should never reach alias nodes as long as it blocks imports.
  _assert_tangent_parse_error(f, 'Import statements')


def test_for():

  def f(x):
    for _ in range(2):
      x += 1
      break
    else:
      x = 0
    return x

  _assert_tangent_parse_error(f, 'Else block')


def test_continue():

  def f(x):
    for _ in range(2):
      continue
    return x

  _assert_tangent_parse_error(f, 'Continue')


def test_lambda():

  def f(_):
    return lambda x: x + 1

  _assert_tangent_parse_error(f, 'Lambda')


def test_yield():

  def f(x):
    yield x + 1

  _assert_tangent_parse_error(f, 'Yield')


def test_global():

  def f(x):
    global testglobal
    testglobal = 0
    return x

  _assert_tangent_parse_error(f, 'Global')


def test_classdef():

  def f(_):

    class Foo(object):
      pass

    return Foo

  _assert_tangent_parse_error(f, 'Class')


if __name__ == '__main__':
  assert not pytest.main([__file__])
