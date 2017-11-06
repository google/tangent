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

from tangent import anf
from tangent import quoting


def anf_lines(f):
  """Return the ANF transformed source code as lines."""
  return quoting.unquote(anf.anf(quoting.parse_function(f))).split('\n')


def anf_function(f, globals_=None):
  m = gast.gast_to_ast(anf.anf(quoting.parse_function(f)))
  m = gast.fix_missing_locations(m)
  exec(compile(m, '<string>', 'exec'), globals_)
  return f


def test_anf():
  def g(x):
    return x * 2

  h = g

  def f(x):
    y = g(h(x))
    return y

  assert anf_lines(f)[1].strip() == "h_x = h(x)"
  assert anf_function(f, locals())(2) == 8

  def f(x):
    return x * x * x

  assert 'return' in anf_lines(f)[-1] and '*' not in anf_lines(f)[-1]
  assert anf_function(f)(2) == 8

  def f(x):
    y = [(x.y[0],), 3]
    y += x * f(x[g(x)].b, (3, x / -2))

  assert anf.anf(quoting.parse_function(f))


def test_long():
  def f(x):
    return some_very_long_name(some_other_long_name(x))

  # If a function name is long, we use the LHS or return statement for the name
  # instead
  assert anf_lines(f)[-1].strip() == 'return _return'
  assert anf_lines(f)[-3].strip().startswith('_return2 = ')

  def f(x):
    some_very_long_variable_name_here = f(some_very_long_function_name(x))
    return some_very_long_variable_name_here

  # If both the target and function name are long, we should back off to short,
  # random variable names
  assert len(anf_lines(f)[-3].strip()) < 40


if __name__ == '__main__':
  assert not pytest.main([__file__])
