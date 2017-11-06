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

from tangent import compile as compile_
from tangent import quoting
from tangent import template


def _wrap(body):
  """Take a list of statements and wrap them in a function to compile."""
  def f():
    pass
  tree = quoting.parse_function(f)
  tree.body[0].body = body
  return tree


def test_variable_replace():
  def f(x):
    x = 2
    return x

  body = template.replace(f, x=gast.Name(id='y', ctx=None, annotation=None))
  assert body[0].targets[0].id == 'y'
  assert isinstance(body[0].targets[0].ctx, gast.Store)
  assert isinstance(body[1].value.ctx, gast.Load)
  compile_.compile_function(_wrap(body))


def test_statement_replace():
  def f(body):
    body

  body = [gast.Expr(value=gast.Name(id=var, ctx=gast.Load(), annotation=None))
          for var in 'xy']
  new_body = template.replace(f, body=body)
  assert len(new_body) == 2
  assert isinstance(new_body[0], gast.Expr)
  compile_.compile_function(_wrap(new_body))


def test_function_replace():
  def f(f, args):
    def f(args):
      pass
  body = template.replace(
    f, f='g', args=[gast.Name(id=arg, ctx=None, annotation=None)
                    for arg in 'ab'])
  assert isinstance(body[0], gast.FunctionDef)
  assert body[0].name == 'g'
  assert len(body[0].args.args) == 2
  assert isinstance(body[0].args.args[0].ctx, gast.Param)
  assert body[0].args.args[1].id == 'b'
  compile_.compile_function(_wrap(body))


def test_partial_gradient_replace():
  def f(x, y):
    d[x] = d[y]

  tree = quoting.parse_function(f)
  transformer = template.ReplaceGradTransformer(template.Replace.PARTIAL)
  new_tree = transformer.visit(tree)
  assert isinstance(new_tree.body[0].body[0].targets[0], gast.Name)
  assert new_tree.body[0].body[0].targets[0].id == '_bx'
  assert new_tree.body[0].body[0].value.id == 'by'
  compile_.compile_function(new_tree)


def test_full_gradient_replace():
  def f(x, y):
    d[x] = d[y]

  tree = quoting.parse_function(f)
  transformer = template.ReplaceGradTransformer(template.Replace.FULL)
  new_tree = transformer.visit(tree)
  assert isinstance(new_tree.body[0].body[0].targets[0], gast.Name)
  assert new_tree.body[0].body[0].targets[0].id == 'bx'
  assert new_tree.body[0].body[0].value.id == 'by'
  compile_.compile_function(new_tree)


def test_node_replace():
  node = template.replace(quoting.quote("a = b"), a="y", b="x * 2")
  assert quoting.unquote(node) == "y = x * 2"


def test_string_replace():
  node = template.replace("a = b", a="y", b="x * 2")
  assert quoting.unquote(node) == "y = x * 2"


if __name__ == '__main__':
  assert not pytest.main([__file__])
