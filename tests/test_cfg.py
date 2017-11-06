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

import tangent
from tangent import annotations as anno
from tangent import cfg


def f(x):
  x
  while True:
    x = x
    x = x
  return x


def g(x):
  if x:
    y = 2
  return x


def h(x, y):
  y = f(x)
  return y


def i(x, y):
  z = h(x, y)
  x = z[0]
  return z


def test_reaching():
  node = tangent.quoting.parse_function(f)
  cfg.forward(node, cfg.ReachingDefinitions())
  body = node.body[0].body
  # Only the argument reaches the expression
  assert len(anno.getanno(body[0], 'definitions_in')) == 1
  while_body = body[1].body
  # x can be either the argument here, or from the previous loop
  assert len(anno.getanno(while_body[0], 'definitions_in')) == 2
  # x can only be the previous line here
  assert len(anno.getanno(while_body[1], 'definitions_in')) == 1
  # x can be the argument here or the last definition from the while body
  assert len(anno.getanno(body[2], 'definitions_in')) == 2


def test_defined():
  node = tangent.quoting.parse_function(g)
  cfg.forward(node, cfg.Defined())
  body = node.body[0].body
  # only x is for sure defined at the end
  assert len(anno.getanno(body[1], 'defined_in')) == 1
  # at the end of the if body both x and y are defined
  if_body = body[0].body
  assert len(anno.getanno(if_body[0], 'defined_out')) == 2


def test_active():
  node = tangent.quoting.parse_function(h)
  cfg.forward(node, cfg.Active(wrt=(1,)))
  body = node.body[0].body
  # y has been overwritten here, so nothing is active anymore
  assert not anno.getanno(body[-1], 'active_out')


def test_active2():
  node = tangent.quoting.parse_function(i)
  cfg.forward(node, cfg.Active(wrt=(1,)))
  body = node.body[0].body
  # through y both x and z are now active
  assert len(anno.getanno(body[-1], 'active_out')) == 3


if __name__ == '__main__':
  assert not pytest.main([__file__])
