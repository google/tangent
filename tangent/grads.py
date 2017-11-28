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
"""Templates for gradient expressions.

The first argument to the adjoint must be the return value of the primal.

Use `d[x]` to denote the gradient of a variable `x`.

If the primal returns a tuple, the first argument to the adjoint is a tuple,
and the adjoint is supposed to define `d[y]` as a tuple.

Templates do not support use of `**kwargs`.

If a keyword argument isn't present in the adjoint, it means that Tangent
doesn't support it, and an error will be raised if it appears in user code.

Adjoints have access to the inputs of the primal, output of the primal, and
gradients with respect to the output. They are expected to contain expressions
for the gradient with respect to the input. They don't have access to any
intermediate variables from the primal.

"""
from __future__ import absolute_import

import math
import types

import gast
import numpy
import tangent
from tangent import tracing


# TODO: Avoid requiring non-differentiables to define @tangent_s.
# All non-differentiable function need to create shadow zero-filled variables
# in forward mode. Currently we achieve that by defining identity @tangent_
# versions of those functions, but a beter approach would be to do that
# automatically.

# Create decorators that add templates to dictionaries
adjoints = {}
primals = {}


def get_module_functions(modules):
  """Finds functions that do not have implemented derivatives.

  Args:
    modules: A list of Python modules. Functions contained in these modules
        will be checked for membership in 'implemented', and if not found,
        will be added to an 'unimplemented' set
    implemented: A Python object containing implemented derivatives. A function
        should be checkable for membership using the `fn in implemented` syntax.

  Returns:
    module_fns: A set of functions, builtins or ufuncs in `modules`.
  """
  module_fns = set()
  for module in modules:
    for key in dir(module):
      attr = getattr(module, key)
      if isinstance(
          attr, (types.BuiltinFunctionType, types.FunctionType, numpy.ufunc)):
        module_fns.add(attr)
  return module_fns


def create_register(dict_):
  def register(key):
    def _(f):
      dict_[key] = f
      return f
    return _
  return register


adjoint = create_register(adjoints)
primal = create_register(primals)


# Functions: f => f, df
@adjoint(gast.FunctionDef)
def dfunction_def(adjoint_body, return_dx):
  def df():
    adjoint_body
    return_dx


# Control flow
@primal(gast.For)
def for_(body, i, iter_, target, push, push_target, _target, _stack, op_id_iter,
         op_id_target):
  i = 0
  for target in iter_:
    _target = target
    i += 1
    body
    push_target(_stack, _target, op_id_target)
  push(_stack, i, op_id_iter)


@adjoint(gast.For)
def dfor_(adjoint_body, i, pop, pop_target, target, _stack, op_id_iter,
          op_id_target):
  i = pop(_stack, op_id_iter)
  for _ in range(i):
    target = pop_target(_stack, op_id_target)
    adjoint_body


@primal(gast.While)
def while_(body, i, test, push, _stack, op_id):
  i = 0
  while test:
    i += 1
    body
  push(_stack, i, op_id)


@adjoint(gast.While)
def dwhile_(adjoint_body, i, pop, _stack, op_id):
  i = pop(_stack, op_id)
  for _ in range(i):
    adjoint_body


@primal(gast.If)
def if_(cond, test, body, orelse, push, _stack, op_id):
  cond = test
  if cond:
    body
  else:
    orelse
  push(_stack, cond, op_id)


@adjoint(gast.If)
def dif_(cond, adjoint_body, adjoint_orelse, pop, _stack, op_id):
  cond = pop(_stack, op_id)
  if cond:
    adjoint_body
  else:
    adjoint_orelse


# Binary ops: z = op(x, y)
@adjoint(gast.Mult)
def mult(z, x, y):
  d[x] = tangent.unbroadcast(d[z] * y, x)
  d[y] = tangent.unbroadcast(d[z] * x, y)


@adjoint(gast.Add)
def add(z, x, y):
  d[x] = tangent.unbroadcast(d[z], x)
  d[y] = tangent.unbroadcast(d[z], y)


@adjoint(gast.Pow)
def pow(z, x, y):
  d[x] = y * x ** (y - 1) * d[z]
  d[y] = numpy.log(x) * x ** y * d[z]


@adjoint(gast.Sub)
def sub(z, x, y):
  d[x] = tangent.unbroadcast(d[z], x)
  d[y] = -tangent.unbroadcast(d[z], y)


@adjoint(gast.Div)
def div(z, x, y):
  d[x] = d[z] / y
  d[y] = -d[z] * x / (y * y)


# Unary ops: y = op(x)
@adjoint(gast.USub)
def usub(y, x):
  d[x] = -d[y]


@adjoint(gast.UAdd)
def uadd(y, x):
  d[x] = d[y]


#
# NumPy adjoints
#


@adjoint(numpy.log)
def log(y, x):
  d[x] = d[y] / x


@adjoint(numpy.cos)
def cos(y, x):
  d[x] = -d[y] * numpy.sin(x)


@adjoint(numpy.sin)
def sin(y, x):
  d[x] = d[y] * numpy.cos(x)


@adjoint(numpy.tan)
def tan(y, x):
  cx = numpy.cos(x)
  d[x] = d[y] / (cx * cx)


@adjoint(numpy.cosh)
def cosh(y, x):
  d[x] = d[y] * numpy.sinh(x)


@adjoint(numpy.sinh)
def sinh(y, x):
  d[x] = d[y] * numpy.cosh(x)


@adjoint(numpy.tanh)
def tanh(y, x):
  d[x] = d[y] * (1.0 - (y * y))


@adjoint(numpy.arccos)
def arccos(y, x):
  d[x] = -d[y] / numpy.sqrt(1.0 - x * x)


@adjoint(numpy.arcsin)
def arcsin(y, x):
  d[x] = d[y] / numpy.sqrt(1.0 - x * x)


@adjoint(numpy.arctan)
def arctan(y, x):
  d[x] = d[y] / (1.0 + x * x)


@adjoint(numpy.exp)
def exp(y, x):
  d[x] = y * d[y]


@adjoint(numpy.sqrt)
def sqrt(y, x):
  d[x] = d[y] / (2.0 * y)


@adjoint(numpy.multiply)
def multiply(z, x, y):
  d[x] = y * d[z]
  d[y] = x * d[z]


@adjoint(numpy.dot)
def dot(y, x1, x2):
  d[x1] = tangent.grad_dot(d[y], x1, x2)
  d[x2] = numpy.transpose(tangent.grad_dot(numpy.transpose(d[y]),
                                           numpy.transpose(x2),
                                           numpy.transpose(x1)))


@adjoint(numpy.atleast_1d)
def atleast_1d(y, x):
  d[x] = numpy.reshape(d[y], numpy.shape(x))


@adjoint(numpy.atleast_2d)
def atleast_2d(y, x):
  d[x] = numpy.reshape(d[y], numpy.shape(x))


@adjoint(numpy.atleast_3d)
def atleast_3d(y, x):
  d[x] = numpy.reshape(d[y], numpy.shape(x))


@adjoint(numpy.reshape)
def reshape(y, x, y_shape):
  d[x] = numpy.reshape(d[y], numpy.shape(x))


@adjoint(numpy.transpose)
def transpose(y, x):
  d[x] = numpy.transpose(d[y])


@adjoint(numpy.broadcast_arrays)
def broadcast_arrays(ys, *args):
  d[args] = tuple(tangent.unbroadcast_to(dy, numpy.shape(arg))
                  for arg, dy in zip(args, d[ys]))


@adjoint(numpy.sum)
def sum(y, x, axis=None, dtype=None, keepdims=False):
  d[x] = tangent.astype(tangent.unreduce(d[y], numpy.shape(x),
                                         axis, keepdims), x)


@adjoint(numpy.mean)
def mean(y, x, axis=None, dtype=None, keepdims=False):
  n = tangent.astype(tangent.array_size(x, axis), x)
  d[x] = tangent.astype(tangent.unreduce(d[y], numpy.shape(x),
                                         axis, keepdims), x) / n


@adjoint(numpy.maximum)
def maximum(ans, x, y):
  d[x] = d[ans] * tangent.balanced_eq(x, ans, y)
  d[y] = d[ans] * tangent.balanced_eq(y, ans, x)


@adjoint(numpy.array)
def aarray(ans,x):
  d[x] = tangent.astype(d[ans],x)


@adjoint(numpy.linalg.det)
def adet(z, x):
  """d|A|/dA = adj(A).T

  See  Jacobi's formula: https://en.wikipedia.org/wiki/Jacobi%27s_formula
  """
  adjugate = numpy.linalg.det(x) * numpy.linalg.pinv(x)
  d[x] = d[z] * numpy.transpose(adjugate)


#
# Tangent adjoints
#


@adjoint(tangent.unreduce)
def aunreduce(y, x, shape, axis, keepdims):
  d[x] = tangent.unbroadcast(d[y], x)


@adjoint(tangent.unbroadcast)
def aunbroadcast(y, x, shape):
  d[x] = tangent.unreduce_like(d[y], x, None, False)


@adjoint(tangent.add_grad)
def aadd_grad(z, left, right):
  d[left] = tangent.unbroadcast(d[z], left)
  d[right] = tangent.unbroadcast(d[z], right)


@adjoint(tangent.astype)
def aastype(z, array, y):
  d[array] = tangent.astype(d[z], array)


@adjoint(tangent.push)
def apush(stack, val, op_id):
  d[val] = tangent.pop(stack, d[op_id])


@adjoint(tangent.pop)
def apop(z, stack, op_id):
  tangent.push(stack, d[z], d[op_id])


@adjoint(tangent.push_stack)
def apush_stack(stack, val, op_id):
  d[val] = tangent.pop_stack(stack, d[op_id])


@adjoint(tangent.pop_stack)
def apop_stack(z, stack, op_id):
  tangent.push_stack(stack, d[z], d[op_id])


@adjoint(tangent.copy)
def acopy(z, x):
  d[x] = tangent.copy(d[z])

#
# Tracing primitives
#


@primal(tracing.Traceable)
def traceable_primal(result, fn, vjp, tmp, args):
  result, vjp = tangent.trace_grad(fn, args)


@adjoint(tracing.Traceable)
def traceable_adjoint(result, vjp, dargs):
  dargs = vjp(d[result])


#
# Blacklist unimplemented NumPy grads
#

# We can enumerate all of the functions that we'd like grads for.
# Until we've written the adjoints of all functions we want to support,
# we will throw an explicit "no grad found" error for those we have not
# finished. UNIMPLEMENTED will contain the list of all of these unimplemented
# grad functions
UNIMPLEMENTED_ADJOINTS = get_module_functions(
    (numpy, numpy.fft, numpy.linalg, numpy.random, math)) - set(adjoints)
