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
"""Fix naive AD rules.

Automatic differentiation proceeds by transforming each statement in isolation.
In principle, this works, but there are some corner cases:

Each variable gets pushed to the stack before being assigned to. However, the
first time a variable get assigned this results in pushing an undefined
variable. We either remove these entirely (`CleanStack`) or we ensure the
variable exists by manually setting the variable to `None` (`FixStack`, e.g.
for loops).

Each partial gets accumulated into the gradient of that variable. The first
time this happens the gradient doesn't exist yet, so we replace accumulation
with assignment (`CleanGrad`) or we explicitly initialize the gradient to zeros
(`FixGrad`, e.g. in loops).

"""
from __future__ import absolute_import
import gast

from tangent import annotations as anno
from tangent import ast as ast_
from tangent import quoting
from tangent import transformers
from tangent import utils


class CleanStack(transformers.TreeTransformer):
  """Remove stack pushes of variables that are never defined."""

  def visit(self, node):
    # Remove all AD-generated pushes of unused variables.
    if anno.hasanno(node, 'push_var') and anno.hasanno(
        node, 'pop') and anno.hasanno(node, 'gen_push'):
      defs = frozenset(id_
                       for id_, node in anno.getanno(node, 'definitions_in'))
      if ast_.get_name(anno.getanno(node, 'push_var')) not in defs:
        self.remove(node)
        self.remove(anno.getanno(node, 'pop'))
    return super(CleanStack, self).visit(node)


class FixStack(transformers.TreeTransformer):
  """Explicitly defines variables that might not be defined."""

  def visit(self, node):
    if anno.hasanno(node, 'push_var'):
      varname = ast_.get_name(anno.getanno(node, 'push_var'))
      if varname not in anno.getanno(node, 'defined_in'):
        self.insert_top(quoting.quote('{} = None'.format(varname)))
    return super(FixStack, self).visit(node)


class CleanGrad(gast.NodeTransformer):
  """Replace `dx = dx + partial` with `dx = partial` if `dx` undefined."""

  def visit_Assign(self, node):
    if isinstance(node.value, gast.Call) and anno.hasanno(node.value.func,
                                                          'add_grad'):
      defs = frozenset(id_ for id_, node in anno.getanno(node,
                                                         'definitions_in'))
      if ast_.get_name(node.targets[0]) not in defs:
        node.value = node.value.args[1]
    return node


class FixGrad(transformers.TreeTransformer):
  """Explicitly initialize gradient to zero if needed."""

  def __init__(self):
    super(FixGrad, self).__init__()
    self.added = set()

  def _init(self, node):
    gradname = ast_.get_name(node)
    if anno.hasanno(node, 'adjoint_var'):
      var = anno.getanno(node, 'adjoint_var')
    else:
      var = anno.getanno(node, 'temp_adjoint_var')
    return gast.Assign(
        targets=[gast.Name(id=gradname, ctx=gast.Store(), annotation=None)],
        value=gast.Call(func=utils.INIT_GRAD, args=[var], keywords=[]))

  def prepend_uninitialized_grads(self, node):
    if anno.hasanno(node, 'defined_in'):
      uses = (succ for succ in gast.walk(node) if
              isinstance(succ, gast.Name) and
              isinstance(succ.ctx, gast.Load))
      for use in uses:
        if ((anno.hasanno(use, 'adjoint_var') or
             anno.hasanno(use, 'temp_adjoint_var')) and
            use.id not in anno.getanno(node, 'defined_in') and
            use.id not in self.added):
          self.added.add(use.id)
          self.insert_top(self._init(use))
    return node

  def visit_Assign(self, node):
    node = self.prepend_uninitialized_grads(node)
    return node

  def visit_AugAssign(self, node):
    node = self.prepend_uninitialized_grads(node)
    return node

  def visit_Expr(self, node):
    node = self.prepend_uninitialized_grads(node)
    return node

  def visit_Return(self, node):
    node = self.prepend_uninitialized_grads(node)
    return node
