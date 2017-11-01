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
"""Functions which perform compiler-style optimizations on the AST."""
from __future__ import absolute_import
from collections import defaultdict
import gast

from tangent import annotate
from tangent import annotations as anno
from tangent import cfg
from tangent import quoting
from tangent import transformers


def fixed_point(f):

  def _fp(node):
    while True:
      before = quoting.to_source(node)
      node = f(node)
      after = quoting.to_source(node)
      if before == after:
        break
    return node

  return _fp


@fixed_point
def optimize(node):
  """Perform a series of optimization passes.

  This function performs a series of optimizations (dead code elimination,
  constant folding, variable folding) on the given AST.
  It optimizes the code repeatedly until reaching a fixed point. The fixed
  point is determine roughly by checking whether the number of lines of
  generated source code changed after the latest pass.

  Args:
    node: The AST to optimize.
  Returns:
    The optimized AST.
  """
  node = dead_code_elimination(node)
  node = constant_folding(node)
  node = assignment_propagation(node)
  return node


@fixed_point
def dead_code_elimination(node):
  """Perform a simple form of dead code elimination on a Python AST.

  This method performs reaching definitions analysis on all function
  definitions. It then looks for the definition of variables that are not used
  elsewhere and removes those definitions.

  This function takes into consideration push and pop statements; if a pop
  statement is removed, it will also try to remove the accompanying push
  statement. Note that this *requires dead code elimination to be performed on
  the primal and adjoint simultaneously*.

  Args:
    node: The AST to optimize.

  Returns:
    The optimized AST.
  """
  to_remove = set(def_[1] for def_ in annotate.unused(node)
                  if not isinstance(def_[1], (gast.arguments, gast.For)))
  for n in list(to_remove):
    for succ in gast.walk(n):
      if anno.getanno(succ, 'push', False):
        to_remove.add(anno.getanno(succ, 'push'))
  transformers.Remove(to_remove).visit(node)
  anno.clearanno(node)
  return node


class ReadCounts(gast.NodeVisitor):
  """Find the number of times that each definition is used.

  Requires `ReachingDefinitions` analysis to have been performed.
  """

  def __init__(self):
    self.n_read = defaultdict(int)

  def visit(self, node):
    if anno.hasanno(node, 'definitions_in'):
      self.reaching_definitions = anno.getanno(node, 'definitions_in')
    super(ReadCounts, self).visit(node)
    if anno.hasanno(node, 'definitions_in'):
      self.reaching_definitions = None

  def visit_Name(self, node):
    if isinstance(node.ctx, gast.Load):
      for def_ in self.reaching_definitions:
        if def_[0] == node.id:
          self.n_read[def_[1]] += 1


def read_counts(node):
  """Check how many times a variable definition was used.

  Args:
    node: An AST to analyze.

  Returns:
    A dictionary from assignment nodes to the number of times the assigned to
        variable was used.
  """
  cfg.forward(node, cfg.ReachingDefinitions())

  rc = ReadCounts()
  rc.visit(node)
  return rc.n_read


@fixed_point
def assignment_propagation(node):
  """Perform assignment propagation.

  Assignment propagation is not a compiler optimization as much as a
  readability optimization. If a variable name is used only once, it gets
  renamed when possible e.g. `y = x; z = y` will become `z = x`.

  Args:
    node: The AST to optimize.

  Returns:
    The optimized AST.
  """
  n_reads = read_counts(node)

  to_remove = []
  for succ in gast.walk(node):
    # We found an assignment of the form a = b
    # - Left-hand side is a Name, right-hand side is a Name.
    if (isinstance(succ, gast.Assign) and isinstance(succ.value, gast.Name) and
        len(succ.targets) == 1 and isinstance(succ.targets[0], gast.Name)):
      rhs_name = succ.value.id
      # We now find all the places that b was defined
      rhs_defs = [def_[1] for def_ in anno.getanno(succ, 'definitions_in')
                  if def_[0] == rhs_name]
      # If b was defined in only one place (not an argument), and wasn't used
      # anywhere else but in a == b, and was defined as b = x, then we can fold
      # the statements
      if (len(rhs_defs) == 1 and isinstance(rhs_defs[0], gast.Assign) and
          n_reads[rhs_defs[0]] == 1 and
          isinstance(rhs_defs[0].value, gast.Name) and
          isinstance(rhs_defs[0].targets[0], gast.Name)):
        # Mark rhs_def for deletion
        to_remove.append(rhs_defs[0])
        # Propagate the definition
        succ.value = rhs_defs[0].value

  # Remove the definitions we folded
  transformers.Remove(to_remove).visit(node)
  anno.clearanno(node)
  return node


class ConstantFolding(gast.NodeTransformer):

  def visit_BinOp(self, node):
    self.generic_visit(node)
    left_val = node.left
    right_val = node.right
    left_is_num = isinstance(left_val, gast.Num)
    right_is_num = isinstance(right_val, gast.Num)

    if isinstance(node.op, gast.Mult):
      if left_is_num and right_is_num:
        return gast.Num(left_val.n * right_val.n)
      if left_is_num:
        if left_val.n == 0:
          return gast.Num(0)
        elif left_val.n == 1:
          return right_val
      if right_is_num:
        if right_val.n == 0:
          return gast.Num(0)
        elif right_val.n == 1:
          return left_val
    elif isinstance(node.op, gast.Add):
      if left_is_num and right_is_num:
        return gast.Num(left_val.n + right_val.n)
      if left_is_num and left_val.n == 0:
        return right_val
      if right_is_num and right_val.n == 0:
        return left_val
    elif isinstance(node.op, gast.Sub):
      if left_is_num and right_is_num:
        return gast.Num(left_val.n - right_val.n)
      if left_is_num and left_val.n == 0:
        return gast.UnaryOp(op=gast.USub(), operand=right_val)
      if right_is_num and right_val.n == 0:
        return left_val
    elif isinstance(node.op, gast.Div):
      if left_is_num and right_is_num:
        return gast.Num(left_val.n / right_val.n)
      if right_is_num and right_val.n == 1:
        return left_val
    elif isinstance(node.op, gast.Pow):
      if left_is_num and right_is_num:
        return gast.Num(left_val.n ** right_val.n)
      if left_is_num:
        if left_val.n == 0:
          return gast.Num(0)
        elif left_val.n == 1:
          return gast.Num(1)
      if right_is_num:
        if right_val.n == 0:
          return gast.Num(1)
        elif right_val.n == 1:
          return left_val
    return node


@fixed_point
def constant_folding(node):
  """Perform constant folding.

  This function also uses arithmetic identities (like multiplying with one or
  adding zero) to simplify statements. However, it doesn't inline constants in
  expressions, so the simplifications don't propagate.

  Args:
    node: The AST to optimize.

  Returns:
    The optimized AST.
  """
  f = ConstantFolding()
  return f.visit(node)
