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
"""Helper functions to create gradient nodes from other nodes."""
from __future__ import absolute_import
from __future__ import division

import gast

from tangent import annotations as anno


def create_grad(node, namer, tangent=False):
  """Given a variable, create a variable for the gradient.

  Args:
    node: A node to create a gradient for, can be a normal variable (`x`) or a
        subscript (`x[i]`).
    namer: The namer object which will determine the name to use for the
        gradient.
    tangent: Whether a tangent (instead of adjoint) is created.

  Returns:
    node: A node representing the gradient with the correct name e.g. the
        gradient of `x[i]` is `dx[i]`.

        Note that this returns an invalid node, with the `ctx` attribute
        missing. It is assumed that this attribute is filled in later.

        Node has an `adjoint_var` annotation referring to the node it is an
        adjoint of.
  """
  if not isinstance(node, (gast.Subscript, gast.Name, gast.Str)):
    raise TypeError

  if anno.hasanno(node, 'temp_var'):
    return create_grad(anno.getanno(node, 'temp_var'), namer, tangent)

  def _name_grad(node):
    if not isinstance(node, gast.Name):
      raise TypeError
    varname = node.id
    name = namer.grad(varname, tangent)
    grad_node = gast.Name(
        id=name, ctx=None, annotation=None)
    anno.setanno(grad_node, 'adjoint_var', node)
    return grad_node
  if isinstance(node, gast.Subscript):
    grad_node = create_grad(node.value, namer, tangent=tangent)
    grad_node.ctx = gast.Load()
    return gast.Subscript(value=grad_node, slice=node.slice, ctx=None)
  elif isinstance(node, gast.Str):
    grad_node = create_grad(
        gast.Name(id=node.s, ctx=None, annotation=None), namer, tangent=tangent)
    return gast.Str(grad_node.id)
  else:
    return _name_grad(node)


def create_temp_grad(node, namer, tangent=False):
  """Create a variable to store partial gradients.

  Args:
    node: See `create_grad`.
    namer: See `create_grad`.
    tangent: See `create_grad`.

  Returns:
    node: See `create_grad`. Returns a node representing the partial gradient.
        Note that this is always a simple variable e.g. the temporary partial
        of `x[i]` can be something like `_dxi`.

        Nodes are given an annotation `temp_adjoint_var`.
  """
  if not isinstance(node, (gast.Subscript, gast.Name)):
    raise TypeError

  def _name_temp_grad(node):
    name = namer.temp_grad(node.id, tangent)
    temp_node = gast.Name(id=name, annotation=None, ctx=None)
    return temp_node
  if isinstance(node, gast.Subscript):
    temp_node = _name_temp_grad(node.value)
  else:
    temp_node = _name_temp_grad(node)
  anno.setanno(temp_node, 'temp_adjoint_var', node)
  return temp_node


def create_temp(node, namer):
  """Create a temporary variable.

  Args:
    node: Create a temporary variable to store this variable in.
    namer: A naming object that guarantees the names are unique.

  Returns:
    node: See `create_grad`. Returns a temporary variable, which is always a
        simple variable annotated with `temp_var`.
  """
  if isinstance(node, gast.Name):
    name = node.id
  elif isinstance(node, (gast.Attribute, gast.Subscript)):
    name = node.value.id
  else:
    raise TypeError
  temp_node = gast.Name(id=namer.temp(name), annotation=None, ctx=None)
  anno.setanno(temp_node, 'temp_var', node)
  return temp_node
