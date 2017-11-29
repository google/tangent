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
"""AST visiting and transformation patterns."""

from __future__ import absolute_import

from collections import deque
from copy import copy

import gast
from tangent import annotations as anno
from tangent import grammar


class TreeTransformer(gast.NodeTransformer):
  """A transformer that allows for non-local changes.

  An extension of the standard `NodeTransformer` in Python's `ast` package.
  This transformer can insert statements right before or after the current
  statement, at the end or beginning of the current block, or at the top of the
  function.

  This class is meant to be subclassed in the same way as Python's
  `NodeTransformer` class. The subclasses can then call the `append`,
  `prepend`, etc. methods as appropriate to transform the AST.

  Note that nodes that are appended or prepended using the `append` and
  `prepend` methods will be visited by the transformer. This means that they
  can recursively append or prepend statements of their own. This doesn't hold
  for statements that are appended/prepended to the block or function body;
  these inserted statements are not visited after being inserted.

  To see which nodes classify as statements or which node fields classify as
  blocks, please see `grammar.py`.

  Attributes:
    to_remove: After the initial pass, this contains a set of nodes that will
        be removed. A second pass is automatically performed using the `Remove`
        transformer to actually remove those nodes.

  """

  def __init__(self):
    self.to_insert = []
    self.to_prepend = []
    self.to_append = []
    self.to_prepend_block = []
    self.to_append_block = []
    self.to_insert_top = deque()
    self.to_remove = set()
    self._top = True

  def prepend(self, node):
    """Prepend a statement to the current statement.

    Note that multiple calls to prepend will result in the last statement to be
    prepended to end up at the top.

    Args:
      node: The statement to prepend.

    Raises:
      ValueError: If the given node is not a statement.

    """
    if not isinstance(node, grammar.STATEMENTS):
      raise ValueError
    self.to_prepend[-1].appendleft(node)

  def append(self, node):
    """Append a statement to the current statement.

    Note that multiple calls to append will result in the last statement to be
    appended to end up at the bottom.

    Args:
      node: The statement to append.

    Raises:
      ValueError: If the given node is not a statement.

    """
    if not isinstance(node, grammar.STATEMENTS):
      raise ValueError
    self.to_append[-1].append(node)

  def remove(self, node):
    """Remove the given node."""
    self.to_remove.add(node)

  def insert_top(self, node):
    """Insert statements at the top of the function body.

    Note that multiple calls to `insert_top` will result in the statements
    being prepended in that order; this is different behavior from `prepend`.

    Args:
      node: The statement to prepend.

    Raises:
      ValueError: If the given node is not a statement.

    """
    if not isinstance(node, grammar.STATEMENTS):
      raise ValueError
    self.to_insert_top.append(node)

  def prepend_block(self, node, reverse=False):
    """Prepend a statement to the current block.

    Args:
      node: The statement to prepend.
      reverse: When called multiple times, this flag determines whether the
          statement should be prepended or appended to the already inserted
          statements.

    Raises:
      ValueError: If the given node is not a statement.

    """
    if not isinstance(node, grammar.STATEMENTS):
      raise ValueError
    if reverse:
      self.to_prepend_block[-1].appendleft(node)
    else:
      self.to_prepend_block[-1].append(node)

  def append_block(self, node, reverse=False):
    """Append a statement to the current block.

    Args:
      node: The statement to prepend.
      reverse: When called multiple times, this flag determines whether the
          statement should be prepended or appended to the already inserted
          statements.

    Raises:
      ValueError: If the given node is not a statement.

    """
    if not isinstance(node, grammar.STATEMENTS):
      raise ValueError
    if reverse:
      self.to_append_block[-1].appendleft(node)
    else:
      self.to_append_block[-1].append(node)

  def visit_statements(self, nodes):
    """Visit a series of nodes in a node body.

    This function is factored out so that it can be called recursively on
    statements that are appended or prepended. This allows e.g. a nested
    expression to prepend a statement, and that statement can prepend a
    statement again, etc.

    Args:
      nodes: A list of statements.

    Returns:
      A list of transformed statements.
    """
    for node in nodes:
      if isinstance(node, gast.AST):
        self.to_prepend.append(deque())
        self.to_append.append(deque())
        node = self.visit(node)
        self.visit_statements(self.to_prepend.pop())
        if isinstance(node, gast.AST):
          self.to_insert[-1].append(node)
        elif node:
          self.to_insert[-1].extend(node)
        self.visit_statements(self.to_append.pop())
      else:
        self.to_insert[-1].append(node)
    return self.to_insert[-1]

  def generic_visit(self, node):
    is_top = False
    if self._top:
      is_top = True
      self._top = False
    for field, old_value in gast.iter_fields(node):
      if isinstance(old_value, list):
        if (type(node), field) in grammar.BLOCKS:
          self.to_prepend_block.append(deque())
          self.to_append_block.append(deque())
          self.to_insert.append(deque())
          new_values = copy(self.visit_statements(old_value))
          self.to_insert.pop()
        else:
          new_values = []
          for value in old_value:
            if isinstance(value, gast.AST):
              value = self.visit(value)
              if value is None:
                continue
              elif not isinstance(value, gast.AST):
                new_values.extend(value)
                continue
            new_values.append(value)
        if isinstance(node, gast.FunctionDef) and field == 'body':
          new_values.extendleft(self.to_insert_top)
          self.to_insert_top = deque([])
        if (type(node), field) in grammar.BLOCKS:
          new_values.extendleft(self.to_prepend_block.pop())
          return_ = None
          if new_values and isinstance(new_values[-1], gast.Return):
            return_ = new_values.pop()
          new_values.extend(self.to_append_block.pop())
          if return_:
            new_values.append(return_)
        old_value[:] = new_values
      elif isinstance(old_value, gast.AST):
        new_node = self.visit(old_value)
        if new_node is None:
          delattr(node, field)
        else:
          setattr(node, field, new_node)
    if is_top and self.to_remove:
      Remove(self.to_remove).visit(node)
    return node


class Remove(gast.NodeTransformer):
  """Remove statements containing given nodes.

  If an entire block was deleted, it will delete the relevant conditional or
  loop entirely. Note that deleting an entire function body will result in an
  invalid AST.

  Calls to user functions that were generated by Tangent will not be removed
  because this might result in incorrect writing and reading from the tape.

  Args:
    to_remove: A set of nodes that need to be removed. Note that the entire
    statement containing this node will be removed e.g. `y = f(x)` with `x`
    being in `to_remove` will result in the entire statement being removed.

  """

  def __init__(self, to_remove):
    self.to_remove = to_remove
    self.remove = False
    self.is_call = False

  def visit(self, node):
    if node in self.to_remove:
      self.remove = True
    if anno.hasanno(node, 'pri_call') or anno.hasanno(node, 'adj_call'):
      # We don't remove function calls for now; removing them also
      # removes the push statements inside of them, but not the
      # corresponding pop statements
      self.is_call = True
    new_node = super(Remove, self).visit(node)
    if isinstance(node, grammar.STATEMENTS):
      if self.remove and not self.is_call:
        new_node = None
      self.remove = self.is_call = False
    if isinstance(node, gast.If) and not node.body:
      # If we optimized away an entire if block, we need to handle that
      if not node.orelse:
        return
      else:
        node.test = gast.UnaryOp(op=gast.Not, operand=node.test)
        node.body, node.orelse = node.orelse, node.body
    elif isinstance(node, (gast.While, gast.For)) and not node.body:
      return node.orelse
    return new_node
