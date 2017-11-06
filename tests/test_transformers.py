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

from tangent import quoting
from tangent import transformers


def test_insert():
  def f(x):
      y = x
      return y
  node = quoting.parse_function(f)

  class Prepend(transformers.TreeTransformer):
    def visit_Assign(self, node):
      # If the target is y, then prepend this statement
      # NOTE Without this test, we'd have an infinite loop
      if node.targets[0].id == 'y':
        statement = quoting.quote("x = 2 * x")
        self.prepend(statement)
      return node

  Prepend().visit(node)
  assert quoting.unquote(node).split('\n')[1].strip() == "x = 2 * x"


def test_insert_block():
  def f(x):
    while True:
      y = x
      z = y
    return y
  node = quoting.parse_function(f)

  class PrependBlock(transformers.TreeTransformer):
    def visit_Assign(self, node):
      # If the target is y, then prepend this statement
      # NOTE Without this test, we'd have an infinite loop
      if node.targets[0].id == 'z':
        statement = quoting.quote("x = 2 * x")
        self.prepend_block(statement)
      return node

  PrependBlock().visit(node)
  assert quoting.unquote(node).split('\n')[2].strip() == "x = 2 * x"


def test_insert_top():
  def f(x):
    while True:
      y = x
      z = y
    return y
  node = quoting.parse_function(f)

  class InsertTop(transformers.TreeTransformer):
    def visit_Assign(self, node):
      # If the target is y, then prepend this statement
      # NOTE Without this test, we'd have an infinite loop
      if node.targets[0].id == 'z':
        statement = quoting.quote("x = 2 * x")
        self.insert_top(statement)
      return node

  InsertTop().visit(node)
  assert quoting.unquote(node).split('\n')[1].strip() == "x = 2 * x"


def test_remove():
  def f(x):
    while True:
      y = x
      z = y
    return y
  node = quoting.parse_function(f)

  class InsertTop(transformers.TreeTransformer):
    def visit_Assign(self, node):
      # If the target is y, then prepend this statement
      # NOTE Without this test, we'd have an infinite loop
      if node.targets[0].id == 'z':
        self.remove(node)
      return node

  InsertTop().visit(node)
  assert quoting.unquote(node).split('\n')[3].strip() == "return y"


if __name__ == '__main__':
  assert not pytest.main([__file__])
