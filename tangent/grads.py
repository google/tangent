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


# TODO(alexbw): test
@adjoint(numpy.radians)
def radians(ans, x):
  d[x] = d[ans] * np.pi / 180.0


@adjoint(numpy.rad2deg)
def rad2deg(ans, x):
  d[x] = d[ans] / numpy.pi * 180.0


@adjoint(numpy.degrees)
def degrees(ans, x):
  d[x] = d[ans] / numpy.pi * 180.0


@adjoint(numpy.deg2rad)
def deg2rad(ans, x):
  d[x] = d[ans] * numpy.pi / 180.0


@adjoint(numpy.square)
def square(ans, x):
  d[x] = d[ans] * 2 * x


@adjoint(numpy.sinc)
def sinc(ans, x):
  d[x] = d[ans] * (numpy.cos(numpy.pi * x) * numpy.pi * x \
                   - numpy.sin(numpy.pi * x)) / (numpy.pi * x * x)


@adjoint(numpy.negative)
def negative(ans, x):
  d[x] = -d[ans]


@adjoint(numpy.abs)
def abs_(ans, x):
  d[x] = d[ans] * tangent.replace_zero(numpy.conj(x),
                                       0.) / tangent.replace_zero(ans, 1.)


@adjoint(numpy.fabs)
def fabs(ans, x):
  d[x] = numpy.sign(x) * d[ans]  # fabs doesn't take complex numbers.


@adjoint(numpy.absolute)
def absolute(ans, x):
  d[x] = d[ans] * numpy.conj(x) / ans


@adjoint(numpy.reciprocal)
def reciprocal(ans, x):
  d[x] = -d[ans] / x**2


@adjoint(numpy.power)
def power(ans, x1, x2):
  d[x1] = tangent.unbroadcast(d[ans] * x2 * x1 ** numpy.where(x2, x2 - 1, 1.0), x1)
  d[x2] = tangent.unbroadcast(d[ans] * numpy.log(tangent.replace_zero(x1, 1.0)) * x1 ** x2, x2)

@adjoint(numpy.arctan2)
def arctan2(ans, x1, x2):
  d[x1] = tangent.unbroadcast(d[ans] * x2 / (x1 ** 2.0 + x2 ** 2.0), x1)
  d[x2] = tangent.unbroadcast(d[ans] * -x1 / (x1 ** 2.0 + x2 ** 2.0), x2)


@adjoint(numpy.hypot)
def hypot(ans, x1, x2):
  d[x1] = tangent.unbroadcast(d[ans] * x1 / ans, x1)
  d[x2] = tangent.unbroadcast(d[ans] * x2 / ans, x2)


@adjoint(numpy.exp)
def exp(ans, x):
  d[x] = ans * d[ans]


@adjoint(numpy.exp2)
def exp2(ans, x):
  d[x] = ans * numpy.log(2) * d[ans]


@adjoint(numpy.expm1)
def expm1(ans, x):
  d[x] = (ans + 1) * d[ans]


@adjoint(numpy.ldexp)
def ldexp(ans, x1, x2):
  d[x1] = tangent.unbroadcast(d[ans] * 2.0**x2, x1)
  d[x2] = tangent.unbroadcast(d[ans] * numpy.log(2) * ans, x2)


@adjoint(numpy.frexp)
def frexp(ans, x):
  dmantissa,dexp = d[ans]
  d[x] = dmantissa / 2 ** numpy.ceil(numpy.log2(numpy.abs(x)))


@adjoint(numpy.log)
def log(ans, x):
  d[x] = d[ans] / x


@adjoint(numpy.log2)
def log2(ans, x):
  d[x] = d[ans] / x / numpy.log(2)


@adjoint(numpy.log10)
def log10(ans, x):
  d[x] = d[ans] / x / numpy.log(10)


@adjoint(numpy.log1p)
def log1p(ans, x):
  d[x] = d[ans] / (x + 1)


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


@adjoint(numpy.arcsinh)
def arcsinh(y, x):
  d[x] = 1 / numpy.sqrt(x**2.0 + 1)


@adjoint(numpy.arccosh)
def arccosh(y, x):
  d[x] = 1 / numpy.sqrt(x**2.0 - 1)


@adjoint(numpy.arctanh)
def arctanh(y, x):
  d[x] = 1. / (1. - x**2.0)


@adjoint(numpy.exp)
def exp(y, x):
  d[x] = y * d[y]


@adjoint(numpy.sqrt)
def sqrt(y, x):
  d[x] = d[y] / (2.0 * y)


@adjoint(numpy.multiply)
def multiply(z, x, y):
  d[x] = tangent.unbroadcast(y * d[z], x)
  d[y] = tangent.unbroadcast(x * d[z], y)


@adjoint(numpy.add)
def add(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans], x)
  d[y] = tangent.unbroadcast(d[ans], y)


@adjoint(numpy.multiply)
def multiply(ans, x, y):
  d[x] = tangent.unbroadcast(y * d[ans], x)
  d[y] = tangent.unbroadcast(x * d[ans], y)


@adjoint(numpy.subtract)
def subtract(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans], x)
  d[y] = tangent.unbroadcast(-d[ans], y)


@adjoint(numpy.divide)
def divide(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans] / y, x)
  d[y] = tangent.unbroadcast(-d[ans] * x / (y * y), y)


@adjoint(numpy.maximum)
def maximum(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(x, ans, y), x)
  d[y] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(y, ans, x), y)


@adjoint(numpy.minimum)
def minimum(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(x, ans, y), x)
  d[y] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(y, ans, x), y)


@adjoint(numpy.fmax)
def fmax(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(x, ans, y), x)
  d[y] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(y, ans, x), y)


@adjoint(numpy.fmin)
def fmin(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(x, ans, y), x)
  d[y] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(y, ans, x), y)


@adjoint(numpy.max)
def max_(ans, x, axis=None, keepdims=None):
  d[x] = tangent.grad_chooser(d[ans], ans, x, axis=axis, keepdims=keepdims)


@adjoint(numpy.min)
def max_(ans, x, axis=None, keepdims=None):
  d[x] = tangent.grad_chooser(d[ans], ans, x, axis=axis, keepdims=keepdims)


@adjoint(numpy.amax)
def max_(ans, x, axis=None, keepdims=None):
  d[x] = tangent.grad_chooser(d[ans], ans, x, axis=axis, keepdims=keepdims)


@adjoint(numpy.amin)
def max_(ans, x, axis=None, keepdims=None):
  d[x] = tangent.grad_chooser(d[ans], ans, x, axis=axis, keepdims=keepdims)


@adjoint(numpy.median)
def median_(ans, x, axis, keepdims=False):
  d[x] = tangent.grad_chooser(d[ans], ans, x, axis=axis, keepdims=keepdims)


@adjoint(numpy.nanmedian)
def nanmedian_(ans, x, axis, keepdims=False):
  d[x] = tangent.grad_chooser(d[ans], ans, x, axis=axis, keepdims=keepdims)


@adjoint(numpy.nanmax)
def nanmax_(ans, x, axis=None, keepdims=None):
  d[x] = tangent.grad_chooser(d[ans], ans, x, axis=axis, keepdims=keepdims)


@adjoint(numpy.nanmin)
def nanmin_(ans, x, axis=None, keepdims=None):
  d[x] = tangent.grad_chooser(d[ans], ans, x, axis=axis, keepdims=keepdims)


@adjoint(numpy.logaddexp)
def logaddexp(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans] * numpy.exp(x - ans), x)
  d[y] = tangent.unbroadcast(d[ans] * numpy.exp(y - ans), y)


@adjoint(numpy.logaddexp2)
def logaddexp2(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans] * 2**(x - ans), x)
  d[y] = tangent.unbroadcast(d[ans] * 2**(y - ans), y)


@adjoint(numpy.true_divide)
def true_divide(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans] / y, x)
  d[y] = tangent.unbroadcast(-d[ans] * x / (y * y), y)


@adjoint(numpy.mod)
def mod(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans], x)
  d[y] = tangent.unbroadcast(-d[ans] * numpy.floor(x / y), y)


@adjoint(numpy.fmod)
def fmod(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans], x)
  d[y] = tangent.unbroadcast(-d[ans] * numpy.round(x / y), y)


@adjoint(numpy.remainder)
def remainder(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans], x)
  d[y] = tangent.unbroadcast(-d[ans] * numpy.floor(x / y), y)


@adjoint(numpy.nan_to_num)
def nan_to_num(ans, x):
  d[x] = numpy.where(numpy.isfinite(x), d[ans], 0.)


@adjoint(numpy.dot)
def dot(y, x1, x2):
  d[x1] = tangent.grad_dot(d[y], x1, x2)
  d[x2] = numpy.transpose(tangent.grad_dot(numpy.transpose(d[y]),
                                           numpy.transpose(x2),
                                           numpy.transpose(x1)))


@adjoint(numpy.cross)
def cross(ans, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
  d[a] = numpy.cross(b, d[ans], axisb, axisc, axisa, axis)
  d[b] = numpy.cross(d[ans], a, axisc, axisa, axisb, axis)


@adjoint(numpy.linspace)
def linspace(ans, start, stop, num):
  d[start] = numpy.dot(numpy.linspace(1.0, 0.0, num), d[ans])
  d[stop] = numpy.dot(numpy.linspace(0.0, 1.0, num), d[ans])


@adjoint(numpy.reshape)
def reshape(y, x, y_shape, order=None):
  d[x] = numpy.reshape(d[y], numpy.shape(x), order=order)


@adjoint(numpy.transpose)
def transpose(y, x):
  d[x] = numpy.transpose(d[y])


@adjoint(numpy.broadcast_arrays)
def broadcast_arrays(ys, *args):
  d[args] = tuple(tangent.unbroadcast_to(dy, numpy.shape(arg))
                  for arg, dy in zip(args, d[ys]))


@adjoint(numpy.cumsum)
def cumsum(ans, a, axis=None):
  d[a] = tangent.grad_cumsum(d[ans], ans, a, axis=axis)


@adjoint(numpy.sum)
def sum_(y, x, axis=None, dtype=None, keepdims=False):
  d[x] = tangent.astype(tangent.unreduce(d[y], numpy.shape(x),
                                         axis, keepdims), x)


@adjoint(numpy.nansum)
def nansum(y, x, axis=None, dtype=None, keepdims=False):
  d[x] = tangent.astype(
      tangent.unreduce(d[y], numpy.shape(x), axis, keepdims), x)


@adjoint(numpy.mean)
def mean(y, x, axis=None, dtype=None, keepdims=False):
  n = tangent.astype(tangent.array_size(x, axis), x)
  d[x] = tangent.astype(tangent.unreduce(d[y], numpy.shape(x),
                                         axis, keepdims), x) / n


@adjoint(numpy.nanmean)
def mean(y, x, axis=None, dtype=None, keepdims=False):
  n = tangent.astype(tangent.array_size(x, axis), x)
  d[x] = tangent.astype(
      tangent.unreduce(d[y], numpy.shape(x), axis, keepdims), x) / n


@adjoint(numpy.var)
def var(ans, x, axis=None, ddof=0, keepdims=False):
  d[x] = tangent.grad_var(
      d[ans], ans, x, axis=axis, ddof=ddof, keepdims=keepdims)


@adjoint(numpy.std)
def std(ans, x, axis=None, ddof=0, keepdims=False):
  d[x] = tangent.grad_std(
      d[ans], ans, x, axis=axis, ddof=ddof, keepdims=keepdims)


@adjoint(numpy.nanvar)
def var(ans, x, axis=None, ddof=0, keepdims=False):
  d[x] = tangent.grad_var(
      d[ans], ans, x, axis=axis, ddof=ddof, keepdims=keepdims)


@adjoint(numpy.nanstd)
def std(ans, x, axis=None, ddof=0, keepdims=False):
  d[x] = tangent.grad_std(
      d[ans], ans, x, axis=axis, ddof=ddof, keepdims=keepdims)


@adjoint(numpy.maximum)
def maximum(ans, x, y):
  d[x] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(x, ans, y), x)
  d[y] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(y, ans, x), y)


@adjoint(numpy.array)
def aarray(ans,x):
  d[x] = tangent.astype(d[ans],x)


@adjoint(numpy.concatenate)
def concatenate(ans, arrays, axis=0):
  d[arrays] = tangent.unconcatenate(d[ans], d[arrays], axis=axis)


# @adjoint(numpy.inner)
# def inner(ans, a, b):
#   pass


@adjoint(numpy.linalg.det)
def adet(z, x):
  """d|A|/dA = adj(A).T

  See  Jacobi's formula: https://en.wikipedia.org/wiki/Jacobi%27s_formula
  """
  adjugate = z * numpy.linalg.pinv(x)
  d[x] = d[z] * numpy.transpose(adjugate)


@adjoint(numpy.clip)
def clip(ans, x, a_min, a_max):
  d[x] = d[ans] * numpy.logical_and(ans != a_min, ans != a_max)


@adjoint(numpy.real_if_close)
def real_if_close(ans, x):
  d[x] = tangent.match_complex(x, d[ans])


@adjoint(numpy.real)
def real(ans, x):
  d[x] = tangent.match_complex(x, d[ans])


@adjoint(numpy.imag)
def imag(ans, x):
  d[x] = tangent.match_complex(x, -1j * d[ans])


@adjoint(numpy.conj)
def conj(ans, x):
  d[x] = numpy.conj(d[ans])


@adjoint(numpy.conjugate)
def conjugate(ans, x):
  d[x] = numpy.conj(d[ans])


@adjoint(numpy.angle)
def angle(ans, x):
  d[x] = tangent.match_complex(x, d[ans] * numpy.conj(x * 1j) / numpy.abs(x)**2)


#
# NumPy shape manipulation adjoints
#


@adjoint(numpy.full)
def full(ans, shape, fill_value):
  d[fill_value] = numpy.sum(d[ans])


@adjoint(numpy.full_like)
def full_like(ans, array_like, fill_value):
  d[fill_value] = numpy.sum(d[ans])


@adjoint(numpy.asarray)
def asarray(ans, array):
  d[array] = tangent.astype(d[ans], array)


@adjoint(numpy.roll)
def roll(ans, a, shift, axis=None):
  d[a] = numpy.roll(d[ans], -shift, axis=axis)


@adjoint(numpy.rollaxis)
def rollaxis(ans, a, axis, start=0):
  if start > axis:
    d[a] = numpy.rollaxis(d[ans], start - 1, axis)
  else:
    d[a] = numpy.rollaxis(d[ans], start, axis + 1)


@adjoint(numpy.moveaxis)
def moveaxis(ans, array, source, destination):
  d[array] = numpy.moveaxis(d[ans], destination, source)


# @adjoint(numpy.take)
# def take(ans, a, indices, axis=None):
#   pass

# @adjoint(numpy.choose)
# def choose(ans, ...):
#   pass


@adjoint(numpy.repeat)
def repeat(ans, x, repeats, axis=None):
  shape = numpy.shape(x)
  if axis == None:
    expanded = numpy.reshape(d[ans], (numpy.prod(shape),) + (repeats,))
  pass


# @adjoint(numpy.put)
# def put(ans, ...):
#   pass


@adjoint(numpy.swapaxes)
def swapaxes(ans, x, axis1, axis2):
  d[x] = numpy.swapaxes(d[ans], axis2, axis1)


# @adjoint(numpy.partition)
# def partition(ans, ...):
#   pass

# @adjoint(numpy.sort)
# def sort(ans, ...):
#   pass

# @adjoint(numpy.resize)
# def resize(ans, ...):
#   pass


@adjoint(numpy.squeeze)
def squeeze(ans, x, axis=None):
  d[x] = numpy.reshape(d[ans], numpy.shape(x))


@adjoint(numpy.diagonal)
def diagonal(ans, a):
  d[a] = tangent.make_diagonal(d[ans])


# @adjoint(numpy.pad)
# def pad(ans, ...):
#   pass


@adjoint(numpy.ravel)
def ravel(ans, a):
  d[a] = numpy.reshape(d[ans], numpy.shape(d[a]))


# @adjoint(numpy.compress)
# def compress(ans, ...):
#   pass

# @adjoint(numpy.fill_diagonal)
# def fill_diagonal(ans, ...):
#   pass


@adjoint(numpy.expand_dims)
def expand_dims(ans, x, axis):
  d[x] = numpy.reshape(d[ans], numpy.shape(x))


# TODO(alexbw): going back to 2D might be problematic, should explicitly reshape!
@adjoint(numpy.column_stack)
def column_stack(ans, tup):
  d[tup] = tangent.unconcatenate(d[ans], d[tup], axis=1)


@adjoint(numpy.row_stack)
def row_stack(ans, tup):
  d[tup] = tangent.unconcatenate(d[ans], d[tup], axis=0)


@adjoint(numpy.array_split)
def array_split(ans, ary, indices_or_sections, axis=0):
  d[ary] = np.concatenate(d[ans], axis=axis)


@adjoint(numpy.split)
def split(ans, ary, indices_or_sections, axis=0):
  d[ary] = np.concatenate(d[ans], axis=axis)


@adjoint(numpy.hsplit)
def hsplit(ans, ary, indices_or_sections):
  d[ary] = np.concatenate(d[ans], axis=1)


@adjoint(numpy.vsplit)
def vsplit(ans, ary, indices_or_sections):
  d[ary] = np.concatenate(d[ans], axis=0)


@adjoint(numpy.dsplit)
def dsplit(ans, ary, indices_or_sections):
  d[ary] = np.concatenate(d[ans], axis=2)


# @adjoint(numpy.kron)
# def kron(ans, ...):
#   pass

# @adjoint(numpy.tile)
# def tile(ans, ...):
#   pass


@adjoint(numpy.atleast_1d)
def atleast_1d(ans, array):
  d[array] = numpy.reshape(d[ans], numpy.shape(d[array]))


@adjoint(numpy.atleast_2d)
def atleast_2d(ans, array):
  d[array] = numpy.reshape(d[ans], numpy.shape(d[array]))


@adjoint(numpy.atleast_3d)
def atleast_3d(ans, array):
  d[array] = numpy.reshape(d[ans], numpy.shape(d[array]))


@adjoint(numpy.vstack)
def vstack(ans, arrays):
  d[arrays] = tangent.unconcatenate(d[ans], d[arrays], axis=0)


@adjoint(numpy.hstack)
def hstack(ans, arrays):
  d[arrays] = tangent.unconcatenate(d[ans], d[arrays], axis=1)


@adjoint(numpy.dstack)
def dstack(ans, arrays):
  d[arrays] = tangent.unconcatenate(d[ans], d[arrays], axis=2)


@adjoint(numpy.stack)
def stack(ans, arrays, axis=0):
  d[arrays] = tangent.unconcatenate(d[ans], d[arrays], axis=axis)


# @adjoint(numpy.trace)
# def trace(ans, ...):
#   pass


@adjoint(numpy.where)
def where(ans, c, x, y):
  d[x] = numpy.where(c, d[ans], numpy.zeros_like(d[ans]))
  d[y] = numpy.where(c, numpy.zeros_like(d[ans]), d[ans])


# @adjoint(numpy.correlate)
# def correlate(ans, ...):
#   pass

# @adjoint(numpy.convolve)
# def convolve(ans, ...):
#   pass


@adjoint(numpy.fliplr)
def fliplr(ans, x):
  d[x] = numpy.fliplr(d[ans])


@adjoint(numpy.flipud)
def flipud(ans, x):
  d[x] = numpy.flipud(d[ans])


@adjoint(numpy.rot90)
def rot90(ans, x, k=1):
  d[x] = numpy.rot90(d[ans], k=-k)


@adjoint(numpy.diag)
def diag(ans, x, k=0):
  d[x] = numpy.diag(d[ans], k)


# @adjoint(numpy.diagflat)
# def diagflat(ans, ...):
#   pass


@adjoint(numpy.tril)
def tril(ans, x, k=0):
  d[x] = numpy.tril(d[ans], k=k)


@adjoint(numpy.triu)
def triu(ans, x, k=0):
  d[x] = numpy.triu(d[ans], k=k)


# @adjoint(numpy.unique)
# def unique(ans, ...):
#   pass

# @adjoint(numpy.intersect1d)
# def intersect1d(ans, ...):
#   pass
#
#
# @adjoint(numpy.setxor1d)
# def setxor1d(ans, ...):
#   pass
#
#
# @adjoint(numpy.union1d)
# def union1d(ans, ...):
#   pass
#
#
# @adjoint(numpy.setdiff1d)
# def setdiff1d(ans, ...):
#   pass
#
#
# @adjoint(numpy.ediff1d)
# def ediff1d(ans, ...):
#   pass
#
#
# @adjoint(numpy.select)
# def select(ans, ...):
#   pass


@adjoint(numpy.copy)
def copy(ans, a):
  d[a] = numpy.copy(d[ans])


# @adjoint(numpy.delete)
# def delete(ans, arr, obj, axis=None):
#   pass

# @adjoint(numpy.insert)
# def insert(ans, ...):
#   pass

# @adjoint(numpy.append)
# def append(ans, ...):
#   pass
#
#
# @adjoint(numpy.extract)
# def extract(ans, ...):
#   pass
#
#
# @adjoint(numpy.trim_zeros)
# def trim_zeros(ans, ...):
#   pass

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
