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
import pytest

from tangent import annotate
from tangent import annotations as anno
from tangent import quoting


def test_resolve():
  def g(x):
    return 2 * x

  def f(x):
    return g(x)

  node = annotate.resolve_calls(f)
  assert anno.getanno(node.body[0].body[0].value, 'func') == g

  def f(x):
    return h(x)

  node = quoting.parse_function(f)
  with pytest.raises(AttributeError):
    annotate.resolve_calls(f)


def test_unused():
  def f(x):
    y = x * 2
    return x

  node = quoting.parse_function(f)
  unused = annotate.unused(node)
  assert unused == set([('y', node.body[0].body[0])])

  def f(x):
    y = x * 2
    return y

  unused = annotate.unused(quoting.parse_function(f))
  assert not unused

  def f(x):
    while True:
      y = x * 2
      x = 3
    return y

  unused = annotate.unused(quoting.parse_function(f))
  assert not unused


if __name__ == '__main__':
  assert not pytest.main([__file__])
