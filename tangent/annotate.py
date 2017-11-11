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
"""Annotate the AST.

This file contains passes that walk the AST and attach annotations to various
nodes.
"""
from __future__ import absolute_import
from collections import defaultdict
import builtins

import gast
import six

from tangent import annotations as anno
from tangent import cfg
from tangent import quoting
from tangent import tracing
from tangent import utils


class ResolveCalls(gast.NodeVisitor):
  """Annotate Call nodes with the function being called."""

  def __init__(self, func):
    self.func = func
    self.namespace = six.get_function_globals(func)
    if six.get_function_closure(func):
      self.namespace.update(dict(zip(
          func.__code__.co_freevars,
          (cell.cell_contents for cell in six.get_function_closure(func)))))

  def visit_FunctionDef(self, node):
    self.generic_visit(node)
    anno.setanno(node, 'func', self.func)

  def visit_Call(self, node):
    self.generic_visit(node)

    def resolve(node):
      if isinstance(node, gast.Attribute):
        return getattr(resolve(node.value), node.attr)
      if isinstance(node, gast.Name):
        if node.id in self.namespace:
          return self.namespace[node.id]
        else:
          # TODO: we should detect when tracing is a fallback.
          if hasattr(builtins, node.id):
            return getattr(builtins, node.id)
          else:
            raise AttributeError(
                'Failed to resolve name "%s" used by "%s".'% (
                    node.id, self.func.__name__))

    func = resolve(node.func)
    # If the user has used the @tangent.trace decorator,
    # then we'll switch to tracing the function.
    if hasattr(func, 'should_trace'):
      func = tracing.Traceable
    elif hasattr(func, 'fun'):
      # TODO: use a less dicey API to check if a function is autograd-wrapped
      # Autograd primitives keep around their original wrapped function.
      # We need that to be the func annotation, otherwise we'd have to
      # redefine derivatives for all autograd wrapped versions of NumPy.
      # Beyond that, autograd wrapped functions only have fn(*args,**kwargs)
      # for their signature. We need access tothe default values of functions
      # for proper code generation.
      func = func.fun
    anno.setanno(node, 'func', func)


def resolve_calls(func):
  """Parse a function into an AST with function calls resolved.

  Since the calls are resolved using the global and local namespace of the
  function it means that procedural parameters (i.e. functions passed as
  arguments) won't be resolved.

  Similarly, functions defined inside of the function that we are trying to
  resolve won't be resolved, since they are not in the local namespace of the
  outer function.

  The function definition itself is also annotated, so that it can be matched
  to calls to it in other functions.

  Args:
    func: The function whose calls are being resolved.

  Returns:
    node: An AST where each `Call` node has a `func` annotation with the
    function handle that the call resolves to.

  Raises:
    AttributeError: When a function is used on the RHS of an assignment cannot
        be resolved (because it was passed as an argument or was defined in the
        body of the function).
  """
  node = quoting.parse_function(func)
  ResolveCalls(func).visit(node)
  return node


def _get_stack_op_handle(node):
  assert isinstance(node, gast.Call), 'Only can get fn handles of Call nodes'
  fn_handle = anno.getanno(node, 'func', False)
  fn_map = defaultdict(lambda: False)
  fn_map['tangent.pop'] = utils.pop
  fn_map['tangent.push'] = utils.push
  fn_map['tangent.pop_stack'] = utils.pop_stack
  fn_map['tangent.push_stack'] = utils.push_stack
  if not fn_handle:
    fn_handle = fn_map[quoting.unquote(node.func)]
  return fn_handle


class FindStackOps(gast.NodeVisitor):
  """Find the pushes and pops in a node, and record all matched op IDs.
  A necessary prerequisite to annotating the push/pop Call and containing
  Assign and Expr nodes in `FindStack`.
  """

  def __init__(self):
    self.push_pop_pairs = dict()

  def visit_Call(self, node):
    fn_handle = _get_stack_op_handle(node)
    if fn_handle and fn_handle in [
        utils.pop, utils.push, utils.push_stack, utils.pop_stack
    ]:
      # Retrieve the op_id, e.g. tangent.push(_stack,val,'abc')
      #                                                   ^^^
      if fn_handle in [utils.push, utils.push_stack]:
        _, _, op_id_node = node.args
      elif fn_handle in [utils.pop, utils.pop_stack]:
        _, op_id_node = node.args
      op_id = op_id_node.s
      if op_id not in self.push_pop_pairs:
        self.push_pop_pairs[op_id] = dict()
      assert fn_handle not in self.push_pop_pairs, (
          'Conflicting op_ids. '
          'Already have fn %s with '
          'op_id %s') % (quoting.unquote(node.func), op_id)
      self.push_pop_pairs[op_id][fn_handle] = node


class AnnotateStacks(gast.NodeVisitor):
  """A class to find pushes and pops to the stack and annotate them as such.

  Args:
    push_pop_pairs: A dict of dicts containing a mapping from op_ids to push/pop
        Call nodes. Compute this using `FindStackOps`.
    strict: A boolean indicating whether to stringently test whether each
        push and pop are matched. This is not always possible when taking
        higher-order derivatives of code generated in split-motion (e.g.
        a function y = primal_f(x) only pushes variables onto a stack for use
        within dx = adjoint_f(dy,x), taking the second-order derivative of the
        call tree that contains these two will only see primal_f in isolation,
        and thus will only see a push, and never the connected pop)

  Push and pop functions are paired using the no-op string argument `op_id`.
  We use these matched strings to annotate the Call nodes, the containing
  Assign (for pop) and Expr (for push) nodes.

  We also track which variables was moved on/off the stack by adding the
  'push_var' and 'pop_var' annotations, which are used in `CleanStack`
  to remove pushes of variables that are never defined.

  Each push Expr is given a 'pop' annotation, pointing to the pop Assign node.
  Each pop Assign is given a 'push' annotation, pointing to the push Expr node.
  """

  def __init__(self, push_pop_pairs, strict):
    self.push_pop_pairs = push_pop_pairs
    self.strict = strict
    self.fn_map = {}
    self.fn_map[utils.pop] = utils.push
    self.fn_map[utils.push] = utils.pop
    self.fn_map[utils.pop_stack] = utils.pop_stack

    self.fn_map[utils.push_stack] = utils.pop_stack

  def visit_Assign(self, node):
    if not isinstance(node.value, gast.Call):
      return
    fn_handle = _get_stack_op_handle(node.value)
    if fn_handle and fn_handle in [utils.pop, utils.pop_stack]:
      # Retrieve the op_id, e.g. val = tangent.pop(_stack,'abc')
      #                                                    ^^^
      _, op_id_node = node.value.args
      op_id = op_id_node.s
      anno.setanno(node, 'pop_var', node.targets[0])

      if op_id not in self.push_pop_pairs:
        raise ValueError('op_id %s not known' % op_id)
      push_pop_nodes = self.push_pop_pairs[op_id]
      keys = push_pop_nodes.keys()
      # Check that the op_id is associated with only two operations
      if self.strict and len(keys) != 2:
        raise ValueError('Instead of 2 push/pop fns, found %d' % len(keys))

      # Make sure that those two operations are either `push` and `pop`
      # or `push_stack` and `pop_stack`.
      if (self.strict and set(keys) != set((utils.push, utils.pop)) and
          set(keys) != set((utils.push_stack, utils.pop_stack))):
        raise ValueError('Invalid push/pop function pair. Found %s' % keys)

      try:
        matching_push = self.push_pop_pairs[op_id][self.fn_map[fn_handle]]
      except KeyError as e:
        if not self.strict:
          return
        else:
          raise e
      anno.setanno(node, 'push', matching_push, False)
      anno.setanno(node.value, 'push', matching_push, False)

  def visit_Expr(self, node):
    if isinstance(node.value, gast.Call):
      fn_handle = _get_stack_op_handle(node.value)
      if fn_handle and fn_handle in [utils.push, utils.push_stack]:
        op_id = node.value.args[-1].s
        anno.setanno(node, 'push_var', node.value.args[1])
        try:
          matching_pop = self.push_pop_pairs[op_id][self.fn_map[fn_handle]]
        except KeyError as e:
          if not self.strict:
            return
          else:
            raise e
        anno.setanno(node, 'pop', matching_pop, False)
        anno.setanno(node.value, 'pop', matching_pop, False)


def find_stacks(node, strict=False):
  """Find pushes and pops to the stack and annotate them as such.

  Args:
    node: An AST node that might contain stack pushes and pops.
    strict: A boolean indicating whether to stringently test whether each
        push and pop are matched. This is not always possible when taking
        higher-order derivatives of code generated in split-motion.

  Returns:
    node: The node passed in, but with pushes and pops annotated in AST nodes.
  """
  # First, find all stack operation IDs.
  fso = FindStackOps()
  fso.visit(node)
  # Using those IDs, make annotations onto the push and pop nodes.
  AnnotateStacks(fso.push_pop_pairs, strict).visit(node)
  return node


class Unused(gast.NodeVisitor):
  """Walks AST to find uses of variable definitions.

  See `unused` for details.
  """

  def __init__(self):
    # A set that contains all the definitions so far
    self.definitions = set()
    # A set of all the definitions potentially used so far
    self.used = set()
    # The definitions that reach the current statement
    self.reaching_definitions = ()

  @property
  def unused(self):
    """Calculate which AST nodes are unused.

    Note that we have to take special care in the case of
    x,y = f(z) where x is used later, but y is not."""
    unused = self.definitions - self.used
    # Filter (variable_name,node) pairs that should be removed, because
    # node is used elsewhere
    used_nodes = set([u[1] for u in self.used])
    unused = set([u for u in unused if u[1] not in used_nodes])
    return unused

  def visit(self, node):
    if anno.hasanno(node, 'definitions_gen'):
      self.definitions.update(anno.getanno(node, 'definitions_gen'))
      self.reaching_definitions = anno.getanno(node, 'definitions_in')
    if isinstance(node, gast.Name) and isinstance(node.ctx, gast.Load):
      self.used.update(def_ for def_ in self.reaching_definitions
                       if def_[0] == node.id)
    super(Unused, self).visit(node)
    if anno.hasanno(node, 'definitions_gen'):
      self.reaching_definitions = None


def unused(node):
  """Find unused definitions that can be remove.

  This runs reaching definitions analysis followed by a walk over the AST to
  find all variable definitions that are not used later on.

  Args:
    node: The AST of e.g. a function body to find unused variable definitions.

  Returns:
    unused: After visiting all the nodes, this attribute contanis a set of
        definitions in the form of `(variable_name, node)` pairs which are
        unused in this AST.
  """
  cfg.forward(node, cfg.ReachingDefinitions())
  unused_obj = Unused()
  unused_obj.visit(node)
  return unused_obj.unused
