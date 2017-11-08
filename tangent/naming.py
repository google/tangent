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
"""Tools for naming conventions."""
from __future__ import absolute_import
import random
import re
import types

import gast
import six

PRIMAL_NAME = 'pri_{}{}'
ADJOINT_NAME = '_d{}d{}'
TANGENT_NAME = '_t{}t{}'
JOINT_NAME = 'd{}d{}'
STACK_NAME = '_stack'
SUBSTACK_NAME = '_substack'


def primal_name(func, wrt):
  """Name for the primal of a function."""
  if not isinstance(func, types.FunctionType):
    raise TypeError(func)
  varnames = six.get_function_code(func).co_varnames
  return PRIMAL_NAME.format(func.__name__, ''.join(varnames[i] for i in wrt))


def _adjoint_name(func, wrt, template):
  if not isinstance(func, types.FunctionType):
    raise TypeError
  varnames = six.get_function_code(func).co_varnames
  return template.format(func.__name__, ''.join(varnames[i] for i in wrt))


def joint_name(func, wrt):
  """Name for a function in joint mode."""
  return _adjoint_name(func, wrt, JOINT_NAME)


def adjoint_name(func, wrt):
  """Name for the adjoint of a function."""
  return _adjoint_name(func, wrt, ADJOINT_NAME)


def tangent_name(func, wrt):
  """Name for a function in forward mode."""
  return _adjoint_name(func, wrt, TANGENT_NAME)


class Names(gast.NodeVisitor):

  def __init__(self):
    self.names = set()

  def visit_Name(self, node):
    if isinstance(node.ctx, (gast.Store, gast.Param)):
      self.names.add(node.id)


def get_names(node):
  """Find the arguments and variables assigned to in a certain node."""
  names = Names()
  names.visit(node)
  return names.names


def uniqify(func):
  """Make sure that a method returns a unique name."""
  @six.wraps(func)
  def unique(self, *args, **kwargs):
    return self.unique(func(self, *args, **kwargs))
  return unique


def uniqify_once(func):
  """Make sure that a method returns a unique name."""
  @six.wraps(func)
  def unique_once(self, *args, **kwargs):
    return self.unique_once(func(self, *args, **kwargs))
  return unique_once


class Namer(object):
  """Generate human-readable names for AST nodes.

  Given an AST node, this class tries to produce a sensible variable name
  that it could be subtituted with.

  In principle, it will try to construct sensible names from the operands and
  operator e.g. `x + y` becomes `x_plus_y`. However, the length of these
  variable names can quickly explode. In that case, we try to back off to using
  the left hand side of the statement if possible e.g. in `z = f(x + y)` the
  expression `x + y` could be named `_z`.

  In case the LHS is not available (because it wasn't given by the calling
  code) or if the LHS name is too long, we fall back to assigning random
  variable names.

  Some methods (such as `grad`) will return the same name when called with the
  same inputs.

  Attributes:
    names: A set of variable names that cannot be used. Allowed to be changed.
    target: The node that is on the LHS of the current statement. Is `None` by
        default. Should be set by the calling code.
  """
  # Naming convention from 'Evaluating Derivatives', b is rev, d is fwd
  ADJOINT_VAR = 'b{}'
  TANGENT_VAR = 'd{}'
  TEMP_VAR = '_{}'
  TEMP_ADJOINT_VAR = '_b{}'
  TEMP_TANGENT_VAR = '_d{}'

  MAX_LENGTH = 15

  def __init__(self):
    self.names = set()
    self.name_mappings = dict()
    # The targets field of the LHS whenever a node inside an assign statement
    # is being named
    self.target = None

  @classmethod
  def build(cls, node):
    """Construct a namer object for a given function scope."""
    if not isinstance(node, gast.FunctionDef):
      raise ValueError
    namer = cls()
    namer.names.update(get_names(node))
    return namer

  def valid(self, name):
    """Ensure a variable name is valid.

    Note: Assumes variable names are ASCII, which isn't necessarily true in
    Python 3.

    Args:
      name: A proposed variable name.

    Returns:
      A valid version of the name.
    """
    name = re.sub('[^0-9a-zA-Z_]', '', name)
    if re.match('[0-9]', name):
      name = '_' + name
    return name

  def trim(self, name):
    """When the name is too long, use the LHS or a random string instead."""
    if len(name) > self.MAX_LENGTH and self.target:
      name = self.TEMP_VAR.format(self._name(self.target))
    if len(name) > self.MAX_LENGTH:
      while True:
        name = '_{:04x}'.format(random.randint(0, 16 ** 4 - 1))
        if name not in self.names:
          break
    return name

  def unique(self, name):
    """Make a variable name unique by appending a number if needed."""
    # Make sure the name is valid
    name = self.valid(name)
    # Make sure it's not too long
    name = self.trim(name)
    # Now make sure it's unique
    unique_name = name
    i = 2
    while unique_name in self.names:
      unique_name = name + str(i)
      i += 1
    self.names.add(unique_name)
    return unique_name

  def unique_once(self, name):
    if name not in self.name_mappings:
      unique_name = self.unique(name)
      self.name_mappings[name] = unique_name
    return self.name_mappings[name]

  def __getattr__(self, attr):
    """Access unwrapped versions of methods.

    Methods are wrapped with `uniqify` to return a unique version of a
    name. Internally the class however might want to use the original
    version of these methods. This method makes those accessible by using a
    leading underscore.
    """
    if attr.startswith('_') and hasattr(self, attr[1:]):
      return getattr(self, attr[1:]).__wrapped__.__get__(self, Namer)
    raise AttributeError

  @uniqify
  def name(self, node):
    namer = getattr(self, 'name_' + node.__class__.__name__)
    return namer(node)

  @uniqify
  def counter(self):
    return 'i'

  @uniqify_once
  def grad(self, name, tangent=False):
    if tangent:
      var_template = self.TANGENT_VAR
    else:
      var_template = self.ADJOINT_VAR
    return var_template.format(name)

  @uniqify
  def temp_grad(self, name, tangent=False):
    if tangent:
      var_template = self.TEMP_TANGENT_VAR
    else:
      var_template = self.TEMP_ADJOINT_VAR
    return var_template.format(name)

  @uniqify_once
  def temp(self, name):
    return self.TEMP_VAR.format(name)

  @uniqify
  def cond(self):
    return 'cond'

  def name_Name(self, node):
    return node.id

  def name_Return(self, node):
    return 'return'

  def name_Tuple(self, node):
    return 't'

  def name_List(self, node):
    return 'l'

  def name_Call(self, node):
    if len(node.args) <= 2:
      return (self._name(node.func) + '_' +
              '_'.join(self._name(arg) for arg in node.args))
    else:
      return self._name(node.func)

  def name_Attribute(self, node):
    return self._name(node.value) + '_' + node.attr

  def name_Subscript(self, node):
    return self._name(node.value) + '_' + self._name(node.slice)

  def name_Index(self, node):
    return self._name(node.value)

  def name_Slice(self, node):
    return ''.join(self._name(i) if i else ''
                   for i in (node.lower, node.upper, node.step))

  def name_ExtSlice(self, node):
    return '_'.join(self._name(d) for d in node.dims)

  def name_Num(self, node):
    num_str = str(node.n)
    num_str = num_str.replace('.', '_')
    num_str = num_str.replace('-', 'neg')
    num_str = num_str.replace('+', 'plus')
    return num_str

  def name_Str(self, node):
    return node.s

  BINOP_NAMES = {
      gast.Add: 'plus',
      gast.Sub: 'minus',
      gast.Mult: 'times',
      gast.Div: 'over',
      gast.FloorDiv: 'intdiv',
      gast.Mod: 'modulo',
      gast.Pow: 'to_the',
      gast.MatMult: 'times'
  }

  def name_BinOp(self, node):
    return '{left}_{op}_{right}'.format(left=self._name(node.left),
                                        right=self._name(node.right),
                                        op=self.BINOP_NAMES[type(node.op)])

  UNARYOP_NAMES = {
      gast.UAdd: 'plus',
      gast.USub: 'minus',
      gast.Not: 'not'
  }

  def name_UnaryOp(self, node):
    return '{op}_{operand}'.format(op=self.UNARYOP_NAMES[type(node.op)],
                                   operand=self._name(node.operand))
