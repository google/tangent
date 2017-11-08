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
"""Transform AST into something similar to A-normal form.

This significantly simplifies certain procedures later on. The ANF
transformations guarantee the following:

All nested expressions on the right hand side of assignments are expanded and
reduced to the following:

  y = x
  y = f(x1, ..., xn)
  z = x + y
  y = -x
  y.i = x
  y = x.i
  y[i] = x
  y = x[i]
  z = x, y

Note that we do not allow tuple unpacking, because statements like `x[i], y =
f(x)` are difficult to process in this case. Hence, unpacking is made explicit.

The value of the return statement is reduced to either a single variable, or a
tuple of variables (nested tuples are expanded).

"""
from __future__ import absolute_import
import gast

from tangent import annotations as anno
from tangent import grammar
from tangent import naming
from tangent import quoting
from tangent import transformers


class ANF(transformers.TreeTransformer):
  """Transform a tree to an ANF-like form."""

  def __init__(self):
    super(ANF, self).__init__()
    # Whether the current statement in question must be trivialized
    self.trivializing = False
    # The original line that is transformed, which is kept as an annotation
    self.src = ''

  def mark(self, node):
    if not anno.hasanno(node, 'pre_anf') and self.src:
      anno.setanno(node, 'pre_anf', self.src)

  def trivialize(self, node):
    if isinstance(node, (gast.Name, type(None)) + grammar.LITERALS):
      return node
    name = self.namer.name(node)
    stmt = gast.Assign(
        targets=[gast.Name(annotation=None, id=name, ctx=gast.Store())],
        value=None)
    self.mark(stmt)
    self.prepend(stmt)
    stmt.value = self.visit(node)
    return gast.Name(annotation=None, id=name, ctx=gast.Load())

  def visit_Call(self, node):
    if self.trivializing:
      for i, arg in enumerate(node.args):
        node.args[i] = self.trivialize(arg)
      for keyword in node.keywords:
        keyword.value = self.trivialize(keyword.value)
    return node

  def visit_FunctionDef(self, node):
    self.namer = naming.Namer.build(node)
    return self.generic_visit(node)

  def visit_BinOp(self, node):
    if self.trivializing:
      node.left = self.trivialize(node.left)
      node.right = self.trivialize(node.right)
    return node

  def visit_UnaryOp(self, node):
    if self.trivializing:
      node.operand = self.trivialize(node.operand)
    return node

  def visit_Return(self, node):
    self.trivializing = True
    self.namer.target = node
    node.value = self.trivialize(node.value)
    self.trivializing = False
    self.namer.target = None
    return node

  def trivialize_slice(self, node):
    if isinstance(node, gast.Slice):
      name = self.namer.name(node)
      target = gast.Name(id=name, ctx=gast.Store(), annotation=None)
      stmt = gast.Assign(targets=[target], value=None)
      self.prepend(stmt)
      stmt.value = gast.Call(
          func=gast.Name(id='slice', ctx=gast.Load(), annotation=None),
          args=[
              self.trivialize(arg) if arg else
              gast.Name(id='None', ctx=gast.Load(), annotation=None)
              for arg in [node.lower, node.upper,
                          node.step]],
          keywords=[])
      return gast.Name(id=name, ctx=gast.Load(), annotation=None)
    elif isinstance(node, gast.ExtSlice):
      name = self.namer.name(node)
      target = gast.Name(id=name, ctx=gast.Store(), annotation=None)
      stmt = gast.Assign(targets=[target], value=None)
      self.prepend(stmt)
      dim_names = [self.trivialize_slice(s).id for s in node.dims]
      stmt.value = gast.Tuple(elts=[
          gast.Name(id=n, ctx=gast.Load(), annotation=None)
          for n in dim_names], ctx=gast.Load())
      return gast.Name(id=name, ctx=gast.Load(), annotation=None)
    elif isinstance(node, gast.Index):
      return self.trivialize(node.value)
    else:
      raise ValueError(node)

  def visit_Subscript(self, node):
    if self.trivializing:
      node.value = self.trivialize(node.value)
      node.slice = gast.Index(value=self.trivialize_slice(node.slice))
    return node

  def visit_Tuple(self, node):
    if self.trivializing:
      node.elts = [self.trivialize(elt) for elt in node.elts]
    return node

  def visit_List(self, node):
    if self.trivializing:
      node.elts = [self.trivialize(elt) for elt in node.elts]
    return node

  def visit_AugAssign(self, node):
    self.trivializing = True
    left = self.trivialize(node.target)
    right = self.trivialize(node.value)
    self.trivializing = False
    node = gast.Assign(targets=[node.target],
                       value=gast.BinOp(left=left, op=node.op, right=right))
    return node

  def visit_Assign(self, node):
    self.src = quoting.unquote(node)
    self.mark(node)
    self.trivializing = True
    self.namer.target = node.targets[0]
    if isinstance(node.targets[0], (gast.Subscript, gast.Attribute)):
      node.value = self.trivialize(node.value)
      node.targets[0] = self.visit(node.targets[0])
    elif isinstance(node.targets[0], gast.Tuple):
      node.value = self.visit(node.value)
      name = self.namer.name(node.targets[0])
      target = gast.Name(id=name, ctx=gast.Store(), annotation=None)
      for i, elt in enumerate(node.targets[0].elts):
        stmt = gast.Assign(
            targets=[elt],
            value=gast.Subscript(
                value=gast.Name(id=name, ctx=gast.Load(),
                                annotation=None),
                slice=gast.Index(value=gast.Num(n=i)),
                ctx=gast.Load()))
        self.mark(stmt)
        self.append(stmt)
      node.targets[0] = target
    elif not isinstance(node.targets[0], gast.Name):
      raise ValueError
    node = self.generic_visit(node)
    self.namer.target = None
    self.trivializing = False
    return node


def anf(node):
  """Turn an AST into ANF-like form."""
  ANF().visit(node)
  return node
