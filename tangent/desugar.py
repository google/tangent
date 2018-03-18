# Copyright 2018 Google Inc.
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

from __future__ import absolute_import

import gast
import copy

from tangent import ast
from tangent import annotations as anno
from tangent import cfg
from tangent import naming
from tangent import quoting
from tangent import template
from tangent import transformers


class ExplicitLoopIndexes(transformers.TreeTransformer):

  def visit_FunctionDef(self, node):
    cfg.forward(node, cfg.Active(range(len(node.args.args))))
    self.namer = naming.Namer.build(node)
    node = self.generic_visit(node)
    return node

  def visit_For(self, node):
    # If the iter is a Name that is active,
    # we need to rewrite the loop.
    # Iterators of the form `for a in x` rely on an implicit
    # indexing operation, which Tangent cannot reverse without
    # more information. So, we will create an explicit
    # indexing operation. Note that we will use
    # integer indexes, which will cause strange behavior if
    # the iterator's `next()` behavior deviates from
    # a plain incrementing index.
    # The right thing to do (eventually) is to write a multiple-dispatch
    # version of the `next` operator, and its adjoint, so that
    # we can handle e.g. dicts.

    if isinstance(node.iter, (gast.Name, gast.Subscript, gast.Attribute)):
      iter_name = ast.get_name(node.iter)
      if iter_name in anno.getanno(node, 'active_in'):
        # for a in x:
        #   f(a)
        # # becomes
        # for i in range(len(x)):
        #   a = x[i]
        #   f(a)

        # Get a unique iterator name
        old_target = copy.deepcopy(node.target)
        new_target = quoting.quote(self.namer.unique('_idx'))
        old_iter = copy.deepcopy(node.iter)

        item_access = template.replace(
          'old_target = x[i]',
          old_target=old_target,
          x=old_iter,
          i=new_target)

        node.target = gast.Name(id=new_target.id, ctx=gast.Store(), annotation=None)
        node.iter = quoting.quote('range(len(%s))' % iter_name)
        anno.setanno(node.iter, 'func', range)
        anno.setanno(node.iter.args[0], 'func', len)
        node.body = [item_access] + node.body

    return node


def explicit_loop_indexes(node):
  node = ExplicitLoopIndexes().visit(node)
  for n in gast.walk(node):
    for key in ('active_in', 'active_out', 'active_gen', 'active_kill'):
      if anno.hasanno(n, key):
        anno.delanno(n, key)
  return node