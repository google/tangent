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
"""Utilities to manipulate the AST and its annotations."""
from __future__ import absolute_import
import copy

import gast

from tangent import annotations as anno
from tangent import quoting
from tangent import utils


def get_name(node):
  """Get the name of a variable.

  Args:
    node: A `Name`, `Subscript` or `Attribute` node.

  Returns:
    The name of the variable e.g. `'x'` for `x`, `x.i` and `x[i]`.
  """
  if isinstance(node, gast.Name):
    return node.id
  elif isinstance(node, (gast.Subscript, gast.Attribute)):
    return get_name(node.value)
  else:
    raise TypeError


def _get_target(node):
  if isinstance(node, (gast.Name, gast.Subscript, gast.Attribute)):
    return set([get_name(node)])
  elif isinstance(node, (gast.Tuple, gast.List)):
    return set.union(*(_get_target(target)
                       for target in node.elts))
  else:
    raise ValueError


def get_updated(node):
  """Return the variable names created or mutated by this statement.

  This function considers assign statements, augmented assign statements, and
  the targets of for loops, as well as function arguments.

  For example, `x[0] = 2` will return `x`, `x, y = 3, 4` will return `x` and
  `y`, `for i in range(x)` will return `i`, etc.

  Args:
    node: An AST node

  Returns:
    A set of variable names (strings) of all the variables created or mutated.
  """
  if isinstance(node, gast.Assign):
    return set.union(*(_get_target(target)
                       for target in node.targets))
  elif isinstance(node, (gast.For, gast.AugAssign)):
    return _get_target(node.target)
  elif isinstance(node, gast.arguments):
    targets = set(arg.id for arg in node.args + node.kwonlyargs)
    if node.vararg:
      targets.add(node.vararg.id)
    if node.kwarg:
      targets.add(node.kwarg.id)
    return targets
  else:
    return set()


def copy_node(node):
  """Copy a node but keep its annotations intact."""
  if not isinstance(node, gast.AST):
    return [copy_node(n) for n in node]
  new_node = copy.deepcopy(node)
  setattr(new_node, anno.ANNOTATION_FIELD,
          getattr(node, anno.ANNOTATION_FIELD, {}).copy())
  return new_node


class ArgAppend(gast.NodeTransformer):
  """Append arguments to a function definition."""

  def __init__(self, node_list):
    self.visited = False
    self.node_list = node_list

  def visit_FunctionDef(self, node):
    if not self.visited:
      node.args.args.extend(self.node_list)
      self.visited = True
    return node


def append_args(node, node_list):
  if not isinstance(node_list, list):
    raise TypeError('Please pass in a list')
  if all([isinstance(n, str) for n in node_list]):
    node_list = [quoting.quote(n) for n in node_list]
  return ArgAppend(node_list).visit(node)


def is_insert_grad_of_statement(node):
  """Check whether a context manager calls `insert_grad_of`.

  Args:
    node: The context manager node.

  Returns:
    Whether or not this node contains `insert_grad_of` calls.

  Raises:
    ValueError: If the `insert_grad_of` calls are mixed with other calls.
  """
  tangent_calls = [anno.getanno(item.context_expr, 'func', None)
                   is utils.insert_grad_of for item in node.items]
  if all(tangent_calls):
    return True
  elif any(tangent_calls):
    raise ValueError
  else:
    return False
