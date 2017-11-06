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
import inspect

import gast
import pytest

from tangent import compile as compile_
from tangent import quoting


def test_compile():
  def f(x):
    return x * 2

  f = compile_.compile_function(quoting.parse_function(f))
  assert f(2) == 4
  assert inspect.getsource(f).split('\n')[0] == 'def f(x):'

  def f(x):
    return y * 2

  f = compile_.compile_function(quoting.parse_function(f), {'y': 3})
  assert f(2) == 6


def test_function_compile():
  with pytest.raises(TypeError):
    compile_.compile_function(quoting.quote('x = y'))
  with pytest.raises(ValueError):
    compile_.compile_function(gast.parse('x = y'))


if __name__ == '__main__':
  assert not pytest.main([__file__])
