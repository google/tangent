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
"""Helper functions and classes for filling in templates.

Functions can be used as templates. In this case, all the variables to be
replaced should be function arguments. This allows static analysis to still
work. For simple templates nodes can be passed as well.

"""
from __future__ import absolute_import

import types
import enum

import gast
import six
from tangent import annotations as anno
from tangent import ast as ast_
from tangent import create
from tangent import naming
from tangent import quoting
from tangent import transformers


class ReplaceTransformer(gast.NodeTransformer):
  """Replace variables with AST nodes.

  The context of the replacements is automatically set to load or store.

  """

  def __init__(self, replacements):
    self.replacements = replacements
    self.seen = set()
    self.is_top = True

  def visit_Expr(self, node):
    if (isinstance(node.value, gast.Name) and
        node.value.id in self.replacements):
      return self.visit(node.value)
    self.generic_visit(node)
    return node

  def visit_FunctionDef(self, node):
    node = self.generic_visit(node)
    if node.name in self.replacements:
      node.name = self.replacements[node.name].id
    return node

  def visit_Name(self, node):
    if node.id in self.replacements:
      # NOTE In principle we don't want to copy, because it might break
      # references held in annotations, but we will copy if we have to to
      # avoid duplicate nodes
      if node.id in self.seen:
        new_nodes = ast_.copy_node(self.replacements[node.id])
      else:
        self.seen.add(node.id)
        new_nodes = self.replacements[node.id]
      if isinstance(new_nodes, gast.AST):
        new_nodes = [new_nodes]
      for new_node in new_nodes:
        anno.setanno(new_node, 'replacement', node, safe=False)
        if 'ctx' in new_node._fields:
          new_node.ctx = node.ctx
      if len(new_nodes) == 1:
        new_nodes, = new_nodes
      return new_nodes
    else:
      return node


Replace = enum.Enum('Replace', ['NONE', 'PARTIAL', 'FULL', 'TANGENT'])


class ReplaceGradTransformer(transformers.TreeTransformer):
  """Interpret the gradient operator `d[x]` in templates.

  The gradient of a temporary variable is the normal gradient i.e. d[_x] =
  dx.

  Args:
    replace_grad: One of the enumerated `Replace` values. If `PARTIAL` then
        `d[x]` will be transformed into the gradient `bx` when read, but
        transformed into a temporary variable (e.g. `_bx`) when written to.
        This ensures that the gradient `bx` doesn't get overwritten if it
        already exists. If the mode is `FULL` then `d[x]` becomes the gradient
        `bx` everywhere. `TANGENT` functions as `FULL` but creates the tangent
        instead of the adjoint i.e. `dx`.
    namer: A `Namer` object which decides on the names to give to the
        gradients. This guarantess temporaries receiving unique names.
    tangent: Whether to create tangents or adjoints i.e. whether we are in
        reverse or forward mode.
  """

  def __init__(self, replace_grad, namer=None, tangent=False):
    self.replace_grad = replace_grad
    if namer is None:
      namer = naming.Namer()
    self.namer = namer

    self.tangent = tangent
    super(ReplaceGradTransformer, self).__init__()

  def visit_Subscript(self, node):
    if isinstance(node.value, (gast.Name, gast.Num)) and node.value.id == 'd':
      if (not isinstance(node.slice, gast.Index) or
          not isinstance(node.slice.value,
                         (gast.Subscript, gast.Name, gast.Str))):
        # This happens when the gradient of a constant is taken
        if self.replace_grad == Replace.TANGENT:
          new_node = gast.Num(0)
        else:
          new_node = gast.Name(id='_', ctx=None, annotation=None)
          self.remove(new_node)
      elif (self.replace_grad in (Replace.FULL, Replace.TANGENT) or
            isinstance(node.ctx, gast.Load)):
        new_node = create.create_grad(node.slice.value, self.namer,
                                      self.tangent)
      elif isinstance(node.ctx, gast.Store):
        new_node = create.create_temp_grad(node.slice.value, self.namer,
                                           self.tangent)
      else:
        raise ValueError
      new_node.ctx = node.ctx
      if isinstance(new_node, gast.Tuple):
        for elt in new_node.elts:
          elt.ctx = node.ctx
      node = new_node
    return node


def replace(template, replace_grad=Replace.PARTIAL,
            namer=None, **replacements):
  """Replace placeholders in a Python template (quote).

  Args:
    template: A function, AST node or string to be used as a template. Note
        that if a function is passed, any placeholder is expected to also be a
        function argument. If a string is passed, it must represent valid
        Python code, and any variable it references is a placeholder.
    replace_grad: If Replace.NONE, statements of the form `d[x]` are ignored.
        For the other possible values, see `ReplaceGradTransformer`.
    namer: See `ReplaceGradTransformer`.
    **replacements: A mapping from placeholder names to (lists of) AST nodes
        that these placeholders will be replaced by. If a string is passed,
        `quote` will be called on it to turn it into a node.

  Returns:
    body: An AST node or list of AST nodes with the replacements made. If the
        template was a function, a list will be returned. If the template was a
        node, the same node will be returned. If the template was a string, an
        AST node will be returned (a `Module` node in the case of a multi-line
        string, an `Expr` node otherwise).

  Raises:
    ValueError: If a function is used as a template and an incorrect set of
        replacements was passed.
  """
  # Handle the 3 different types of templates: funcs, nodes, and strings
  is_function = isinstance(template, types.FunctionType)
  if is_function:
    tree = quoting.parse_function(template).body[0]
    placeholders = set(arg.id for arg in tree.args.args)
    tree.args.args = []
    if tree.args.vararg:
      placeholders.add(tree.args.vararg)
      tree.args.vararg = None
    if set(replacements.keys()) != placeholders:
      raise ValueError('too many or few replacements')
  elif isinstance(template, gast.AST):
    tree = template
  else:
    tree = quoting.quote(template, return_expr=True)
  # If the replacements are strings, turn them into nodes
  for k, v in replacements.items():
    if isinstance(v, six.string_types):
      replacements[k] = quoting.quote(v)
  # Perform the replacement
  ReplaceTransformer(replacements).visit(tree)
  # Handle the d[x] operator
  if replace_grad is not Replace.NONE:
    rgt = ReplaceGradTransformer(
        replace_grad=replace_grad,
        namer=namer,
        tangent=replace_grad is Replace.TANGENT)
    rgt.visit(tree)
  # Return the AST node with replacements made
  if is_function:
    return tree.body
  else:
    return tree
