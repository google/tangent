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
"""Templates for tangent expressions.

The first argument to the tangent must be the return value of the primal.

Use `d[x]` to denote the derivative of a variable `x`.

If the primal returns a tuple, the first argument to the tangent is a tuple,
and the adjoint is supposed to define `d[y]` as a tuple.

Templates do not support use of `**kwargs`.

If a keyword argument isn't present in the tangent compound statements, it means
that Tangent doesn't support it, and an error will be raised if it appears in
user code.

Tangents have access to the inputs and outputs of the primal. They are expected
to contain expressions for the derivative with respect to the output. They don't
have access to any intermediate variables from the primal.
"""
from __future__ import absolute_import

import math

import gast
import numpy
import tangent
from tangent import grads

tangents = {}
tangent_ = grads.create_register(tangents)


#
# AST tangents
#


@tangent_(gast.Assign)
def tassign(temp, tangent, target, value):
  temp = value
  tangent
  target = temp


@tangent_(gast.Num)
def tnum(z, x):
  d[z] = tangent.init_grad(x)


@tangent_(gast.Name)
def tname(z, x):
  d[z] = d[x]


@tangent_(gast.Attribute)
def tattr(z, x):
  d[z] = tangent.init_grad(x)


@tangent_(gast.Subscript)
def tsubscript(z, x):
  d[z] = d[x]


# For a reference for primitive tangents, see:
# https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers
# or
# https://en.wikipedia.org/wiki/Differentiation_rules
# Note that we don't use "dual numbers", that's a data structure that's useful
# for doing run-time forward-mode automatic differentiation. We're doing
# compile-time autodiff, and we can keep track of the directional derivatives
# in individual variables, with no need to store them alongside the
# original values.


@tangent_(gast.Add)
def tadd(z, x, y):
  d[z] = d[x] + d[y]


@tangent_(gast.Mult)
def tmult(z, x, y):
  d[z] = d[x] * y + x * d[y]


@tangent_(gast.Sub)
def tsub(z, x, y):
  d[z] = d[x] - d[y]


@tangent_(gast.Div)
def tdiv(z, x, y):
  d[z] = (d[x] * y - x * d[y]) / (y * y)


@tangent_(gast.Pow)
def tpow(z, x, y):
  d[z] = y * (x ** (y - 1.0)) * d[x]


@tangent_(gast.USub)
def tusub(z, x):
  d[z] = -d[x]


#
# Collection tangents
#


@tangent_(tuple)
def ttangent(z, x):
  d[z] = tuple(d[x])


@tangent_(list)
def tlist(z, x):
  d[z] = list(d[x])


#
# NumPy tangents
#


@tangent_(numpy.cos)
def tcos(z, x):
  d[z] = -d[x] * numpy.sin(x)


@tangent_(numpy.sin)
def tsin(z, x):
  d[z] = d[x] * numpy.cos(x)


@tangent_(numpy.tan)
def ttan(z, x):
  cx = numpy.cos(x)
  d[z] = d[x] / (cx * cx)


@tangent_(numpy.cosh)
def tcosh(z, x):
  d[z] = d[x] * numpy.sinh(x)


@tangent_(numpy.sinh)
def tsinh(z, x):
  d[z] = d[x] * numpy.cosh(x)


@tangent_(numpy.tanh)
def ttanh(z, x):
  cx = numpy.cosh(x)
  d[z] = d[x] / (cx * cx)


@tangent_(numpy.arccos)
def tarccos(z, x):
  d[z] = -d[x] / numpy.sqrt(1.0 - x * x)


@tangent_(numpy.arcsin)
def tarcsin(z, x):
  d[z] = d[x] / numpy.sqrt(1.0 - x * x)


@tangent_(numpy.arctan)
def tarctan(z, x):
  d[z] = d[x] / (1.0 + x * x)


@tangent_(numpy.exp)
def texp(z, x):
  d[z] = d[x] * z


@tangent_(numpy.log)
def tlog(z, x):
  d[z] = d[x] / x


@tangent_(numpy.sqrt)
def tsqrt(z, x):
  d[z] = d[x] / (2 * z)


@tangent_(numpy.dot)
def tdot(z, x, y):
  d[z] = numpy.dot(d[x], y) + numpy.dot(x, d[y])


@tangent_(numpy.atleast_1d)
def tatleast_1d(z, x):
  d[z] = numpy.atleast_1d(d[x])


@tangent_(numpy.atleast_2d)
def tatleast_2d(z, x):
  d[z] = numpy.atleast_2d(d[x])


@tangent_(numpy.atleast_3d)
def tatleast_3d(z, x):
  d[z] = numpy.atleast_3d(d[x])


@tangent_(numpy.transpose)
def ttranspose(z, x):
  d[z] = numpy.transpose(d[x])


@tangent_(numpy.sum)
def tsum(y, x, axis=None, dtype=None, keepdims=False):
  d[y] = numpy.sum(d[x], axis=axis, dtype=dtype, keepdims=keepdims)


@tangent_(numpy.mean)
def tmean(
    y, x, axis=None, dtype=None, keepdims=False):
  d[y] = numpy.mean(d[x], axis=axis, dtype=dtype, keepdims=keepdims)


@tangent_(numpy.multiply)
def tmultiply(z, x, y):
  d[z] = numpy.multiply(d[x], y) + numpy.multiply(x, d[y])


@tangent_(numpy.arange)
def tarange(z, stop):
  d[z] = numpy.zeros_like(z)


@tangent_(numpy.ndim)
def tndim(z, x):
  d[z] = numpy.ndim(d[x])


@tangent_(numpy.rollaxis)
def trollaxis(z, a, axis, start=0):
  d[z] = numpy.rollaxis(d[a], axis, start)


@tangent_(numpy.shape)
def tshape(z, x):
  d[z] = numpy.shape(d[x])


@tangent_(numpy.array)
def tarray(z, x):
  d[z] = numpy.array(d[x])


#
# Tangent tangents
#


@tangent_(tangent.add_grad)
def tadd_grad(z, x, y):
  d[z] = tangent.add_grad(d[x], d[y])


@tangent_(tangent.init_grad)
def tinit_grad(z, x, allow_lazy_initializer=False):
  d[z] = tangent.init_grad(d[x], allow_lazy_initializer=False)


@tangent_(tangent.push)
def tpush(x, stack, op_id):
  tangent.push(d[stack], d[x], d[op_id])


@tangent_(tangent.push_stack)
def tpush_stack(x, stack, op_id):
  tangent.push_stack(d[stack], d[x], d[op_id])


@tangent_(tangent.pop)
def tpop(x, stack, op_id):
  d[x] = tangent.pop(d[stack], d[op_id])


@tangent_(tangent.pop_stack)
def tpop_stack(x, stack, op_id):
  d[x] = tangent.pop_stack(d[stack], d[op_id])


@tangent_(tangent.unbroadcast)
def tunbroadcast(z, x, y):
  d[z] = tangent.unbroadcast(d[x], d[y])


@tangent_(tangent.Stack)
def tstack(z):
  d[z] = tangent.Stack()


@tangent_(tangent.astype)
def tastype(z, x, y):
  d[z] = tangent.astype(d[x], d[y])


@tangent_(tangent.unreduce)
def tunreduce(z, array, shape, axis, keepdims):
  d[z] = tangent.unreduce(d[array], d[shape], axis, keepdims)



# Until we've written the adjoints of all functions we want to support,
# we will throw an explicit "no tangent found" error for those we have not
# finished. UNIMPLEMENTED will contain the list of all of these unimplemented
# tangent functions

UNIMPLEMENTED_TANGENTS = grads.get_module_functions(
    (numpy, numpy.fft, numpy.linalg, numpy.random, math)) - set(tangents)
