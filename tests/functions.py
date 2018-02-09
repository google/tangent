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
"""Functions to test.

Notes
-----
Functions can have up to 3 scalar arguments with names a, b and c. If the
function has an optional integer argument it must be called n. Vector inputs
have the name x.

"""
from __future__ import division

import numpy as np

import tangent
from tangent import insert_grad_of

import tensorflow as tf


def id_(a):
  return a


def unary_sub(a):
  return -a


def sum_(x):
  return np.sum(x)


def cast_sum_(x):
  return np.sum(np.array(x))


def overwrite_call(x):
  x = np.sum(x)
  x = x * x
  return x


def tanh(a):
  return np.tanh(a)


def identity_assign(a):
  b = a
  return b


def increment(a):
  return a + 1


def saxpy(a, b, c):
  return a * b + c


def mul(a, b):
  return a * b


def add(a, b):
  return a + b


def saxpy_call(a, b, c):
  return add(mul(a, b), c)


def saxpy_anf(a, b, c):
  t1 = a * b
  t2 = t1 + c
  return t2


def saxpy_overwrite(a, b, c):
  t = a * b
  t = t + c
  return t


def gradname_aliasing(a):
  da = a * a
  return da


def overwrite_if(a):
  if a > 0:
    a = a * 3.0
  else:
    a = a * 2.0
  return a


def test_binop_mul1(a):
  return a * 3.0


def test_binop_mul2(a):
  return 3.0 * a


def test_binop_mul3(a):
  return a * a


def test_binop_div1(a):
  return a / 3.0


def test_binop_div2(a):
  return 3.0 / a


def test_binop_div3(a):
  return a / a


def test_binop_sub1(a):
  return a - 3.0


def test_binop_sub2(a):
  return 3.0 - a


def test_binop_sub3(a):
  return a - a


def test_binop_add1(a):
  return a + 3.0


def test_binop_add2(a):
  return 3.0 + a


def test_binop_add3(a):
  return a + a


def nested_if(a):
  if a > 0:
    if a < 10:
      a = a * a
  else:
    a = 3.0 * a
  return a


def multiarg_if(a):
  # A bunch of spammy nonsense to try to break our system
  if a * a / 3.0 > 0:
    b = 3.0
  else:
    b = 4.0
  a = a * b
  if a < b - a * b:
    a = a * a
  elif a > b + 3.0 * a:
    if a / 2.0 < 0:
      a = a / b
    else:
      a = a * b
  else:
    a = -a
  return a


def fn_multiply(a):
  return np.multiply(a, 3.0)


def fn_multiple_return(a):
  return 2 * a, a


def test_anf_list(a):
  b = [1, a * a, 2]
  return np.sum(b)


def test_deep_anf_list(x):
  b = [1, x[0] * x[1] * x[2], 3]
  return np.sum(b)

  # TODO: needs equivalent for all iterable collections


def test_subscript1(x):
  a = x[0]
  b = x[1]
  return a * b


def test_subscript2(x):
  a = x[0]
  b = x[1]
  x[0] = a * b
  return np.sum(x)


def test_subscript3(x):
  y = x ** 2.0
  x[0] = y[1] * y[2]
  return np.sum(x * y)


def test_list_and_subscript(a):
  x = [1.0, a, 3.0]
  return x[0] * x[1]

# TODO: needs a numpy equivalent, and all iterables collections too
# def test_subscript_overwrite(a):
#     x = [1,a*a,3]
#     x[1] = x[0]*x[1]
#     return x[1]+x[1]


def serial_if(a):
  if a > 0:
    a = a * a
  a = a + a
  if a < 0:
    a = 3.0 * a
  return a


def multivar_if(a, b):
  if a > b:
    a = a * a
  else:
    a = a * b
  return a

# TODO: split this into a bunch of tinier tests


def serial_for(a, n):
  for i in range(n):
    a = a * a
  a = 2.0 * a
  for i in range(n + 1):
    a = a * 3.0
  return a


def serial_ifthenfor(a, n):
  if a > 0:
    a = a * a
  a = 2.0 * a
  for i in range(n):
    a = a * a
  return a


def serial_forthenif(a, n):
  for i in range(n):
    a = a * a
  a = 2.0 * a
  if a > 0:
    a = a * a
  return a


def devilish_nested_if(a):
  if a > 0:
    a = a * 3.0
    if a < 10:
      a = a * a
    else:
      a = a * 2.0
  return a


def overwrite_ifelse(a):
  if a > 0:
    a = a * 3.0
  elif 0 > a:
    a = a * 2.0
  else:
    a = 1.0 * a
  return a


def overwrite_arg(a):
  a = a * 3.0
  return a


def overwrite_non_arg(a):
  b = a
  b = b * 3.0
  return b


def third_pow(a):
  return a * a * a


def direct_third_pow(a):
  return a ** 3


def iter_third_pow1(a):
  out = 1
  for i in range(3):
    out = out * a
  return out


def iter_third_pow2(a):
  out = a
  for i in range(3 - 1):
    out = out * a
  return out


def iterpower_static(a):
  for i in range(3):
    a = a * a
  return a


def iterpower(a, n):
  for i in range(n):
    a = a * a
  return a


def cond_iterpower1(a):
  for i in range(3):
    if a < 20:
      a = a * a
  return a


def cond_iterpower2(a):
  if a < 20:
    for i in range(3):
      a = a * a
  return a


def superfor_iterpower(a):
  for i in range(2):
    for j in range(3):
      a = a * a
  return a


def super_iterpower(a):
  # Tests ANF in a for loop
  for i in range(3):
    a = a * a * a
  return a

# ================================================
# Numpy grads
# ================================================


def numpy_sum(x):
  return np.sum(x)


def numpy_mean(x):
  return np.mean(x)


def numpy_var(x):
  return np.var(x)


def numpy_std(x):
  return np.std(x)


def numpy_nanvar(x):
  return np.nanvar(x)


def numpy_nanstd(x):
  return np.nanstd(x)


def numpy_exp(a):
  return np.exp(a)


def numpy_exp2(a):
  return np.exp(np.exp(a))


def numpy_expm1(a):
  return np.expm1(a)


def numpy_ldexp(a, n):
  return np.ldexp(a, n)


def numpy_frexp(x):
  mantissa, exponent = np.frexp(x)
  return np.sum(mantissa)


def numpy_log(a):
  return np.log(a)


def numpy_log2(a):
  return np.log2(a)


def numpy_log10(a):
  return np.log10(a)


def numpy_log1p(a):
  return np.log1p(a)


def numpy_sqrt(a):
  if a >= 0:
    r = np.sqrt(a)
  else:
    r = np.sqrt(-a)
  return r


def numpy_negative(a):
  return np.negative(a)


def numpy_reciprocal(a):
  return np.reciprocal(a)


def numpy_add(a, b):
  return np.add(a, b)


def numpy_logaddexp(a, b):
  return np.logaddexp(a, b)


def numpy_logaddexp2(a, b):
  return np.logaddexp2(a, b)


def numpy_subtract(a, b):
  return np.subtract(a, b)


def numpy_divide(a, b):
  return np.divide(a, b)


def numpy_true_divide(a, b):
  return np.true_divide(a, b)


def numpy_mod(a, b):
  return np.mod(a, b)


def numpy_fmod(a, b):
  return np.fmod(a, b)


def numpy_remainder(a, b):
  return np.remainder(a, b)


def numpy_cross(x1, x2):
  a = x1[:3]
  b = x2[:3]
  return np.sum(np.cross(a, b))


def numpy_linspace(a):
  return np.sum(np.linspace(0, a, 100))


def numpy_cumsum(x):
  return np.sum(np.cumsum(x))


def numpy_cos(a):
  return np.cos(a)


def numpy_sin(a):
  return np.sin(a)


def numpy_tan(a):
  return np.tan(a)


def numpy_cosh(a):
  return np.cosh(a)


def numpy_sinh(a):
  return np.sinh(a)


def numpy_tanh(a):
  return np.tanh(a)


def numpy_arccos(x):
  return np.sum(np.arccos(np.clip(x,-1,1)))


def numpy_arcsin(x):
  return np.sum(np.arcsin(np.clip(x,-1,1)))


def numpy_sinc(a):
  return np.sinc(a)


def numpy_arctan(a):
  return np.arctan(a)


def numpy_arcsinh(a):
  return np.arcsinh(a)


def numpy_arccosh(a):
  return np.arccosh(a)


def numpy_arctanh(x):
  return np.sum(np.arctanh(np.clip(x,-1,1)))


def numpy_radians(a):
  return np.radians(a)


def numpy_rad2deg(a):
  return np.rad2deg(a)


def numpy_degrees(a):
  return np.degrees(a)


def numpy_deg2rad(a):
  return np.deg2rad(a)


def numpy_square(a):
  return np.square(a)


def numpy_abs(x):
  return np.sum(np.abs(x))


def numpy_fabs(x):
  return np.sum(np.fabs(x))


# TODO(alexbw): test with complex values
def numpy_absolute(x):
  return np.sum(np.absolute(x))


def numpy_arctan2(a, b):
  return np.arctan2(a, b)


def numpy_hypot(a, b):
  return np.hypot(a, b)


def numpy_maximum(x1, x2):
  return np.maximum(x1, x2)


def numpy_minimum(x1, x2):
  return np.minimum(x1, x2)


def numpy_fmax(x1, x2):
  return np.fmax(x1, x2)


def numpy_fmin(x1, x2):
  return np.fmin(x1, x2)


def numpy_max1(sqm):
  return np.max(sqm)


def numpy_max2(sqm):
  return np.sum(np.max(sqm, axis=0))


def numpy_min1(sqm):
  return np.min(sqm)


def numpy_min2(sqm):
  return np.sum(np.min(sqm, axis=0))


def numpy_median1(sqm):
  return np.median(sqm)


def numpy_median2(sqm):
  return np.sum(np.median(sqm, axis=0))


# def numpy_inner(x):
#   return np.sum(np.inner(x, x))


def numpy_full(a):
  return np.sum(np.full((3, 3), a))


def numpy_full_like(a):
  array = np.eye(3)
  return np.sum(np.full_like(array, a))


def numpy_asarray(x):
  return np.sum(np.array(x))


def numpy_roll1(x):
  return np.sum(np.roll(x, 1))


def numpy_roll2(x):
  return np.roll(x, 1)[0]


def numpy_rollaxis(sqm):
  return np.sum(np.rollaxis(sqm, 1, 0)[0])


def numpy_moveaxis(sqm):
  return np.sum(np.moveaxis(sqm, 1, 0)[0])


# def numpy_take(...):
#   return np.take(...)
#
#
# def numpy_choose(...):
#   return np.choose(...)
#
#
def numpy_repeat(sqm):
  return np.sum(np.repeat(sqm, 3, 0))


#
#
# def numpy_put(...):
#   return np.put(...)


def numpy_swapaxes(sqm):
  return np.sum(np.swapaxes(sqm, 1, 0))


# def numpy_partition(...):
#   return np.partition(...)
#
#
# def numpy_sort(...):
#   return np.sort(...)
#
#
# def numpy_resize(...):
#   return np.resize(...)


def numpy_squeeze(x):
  return np.sum(np.squeeze(np.atleast_2d(x)))


def numpy_diagonal(sqm):
  return np.sum(np.diagonal(sqm))


# def numpy_pad(...):
#   return np.pad(...)


def numpy_ravel(sqm):
  return np.sum(np.ravel(sqm))


# def numpy_compress(...):
#   return np.compress(...)
#
#
# def numpy_fill_diagonal(...):
#   return np.fill_diagonal(...)


def numpy_expand_dims(x):
  return np.sum(np.expand_dims(x, 1))


def numpy_column_stack(x):
  return np.sum(np.column_stack((x, x)))


def numpy_row_stack(x):
  return np.sum(np.row_stack((x, x)))


def numpy_array_split(sqm):
  return np.sum(np.array_split(sqm, 3)[0])


def numpy_split(sqm):
  return np.sum(np.split(sqm, 3)[0])


def numpy_hsplit(sqm):
  return np.sum(np.hsplit(sqm, 3)[0])


def numpy_vsplit(sqm):
  return np.sum(np.vsplit(sqm, 3)[0])


def numpy_dsplit(sqm):
  m = np.swapaxes(np.atleast_3d(sqm), 2, 0)
  return np.sum(np.dsplit(m, 3)[0])


# def numpy_kron(...):
#   return np.kron(...)
#
#
# def numpy_tile(...):
#   return np.tile(...)
#


def numpy_atleast_1d(sqm):
  return np.sum(np.atleast_1d(sqm))


def numpy_atleast_2d(sqm):
  return np.sum(np.atleast_2d(sqm))


def numpy_atleast_3d(sqm):
  return np.sum(np.atleast_3d(sqm))


def numpy_concatenate(sqm):
  return np.sum(np.concatenate([sqm, sqm]))


def numpy_dstack(sqm):
  return np.sum(np.dstack([sqm, sqm]))


def numpy_hstack(sqm):
  return np.sum(np.hstack([sqm, sqm]))


def numpy_vstack(sqm):
  return np.sum(np.vstack([sqm, sqm]))


def numpy_stack(sqm):
  return np.sum(np.stack([sqm, sqm]))


# def numpy_trace(...):
#   return np.trace(...)


def numpy_where(x1, x2):
  return np.sum(np.where(x1 > 0, x1, x2))


# def numpy_correlate(...):
#   return np.correlate(...)
#
#
# def numpy_convolve(...):
#   return np.convolve(...)


def numpy_fliplr(sqm):
  return np.sum(np.fliplr(sqm))


def numpy_flipud(sqm):
  return np.sum(np.flipud(sqm))


def numpy_rot90(sqm):
  return np.sum(np.rot90(sqm))


def numpy_diag(sqm):
  return np.sum(np.diag(sqm))


# def numpy_diagflat(...):
#   return np.diagflat(...)


def numpy_tril(sqm):
  return np.sum(np.tril(sqm))


def numpy_triu(sqm):
  return np.sum(np.triu(sqm))


# def numpy_unique(...):
#   return np.unique(...)
#
#
# def numpy_intersect1d(...):
#   return np.intersect1d(...)
#
#
# def numpy_setxor1d(...):
#   return np.setxor1d(...)
#
#
# def numpy_union1d(...):
#   return np.union1d(...)
#
#
# def numpy_setdiff1d(...):
#   return np.setdiff1d(...)
#
#
# def numpy_ediff1d(...):
#   return np.ediff1d(...)
#
#
# def numpy_select(...):
#   return np.select(...)
#
#
# def numpy_copy(...):
#   return np.copy(...)
#
#
# def numpy_delete(...):
#   return np.delete(...)
#
#
# def numpy_insert(...):
#   return np.insert(...)
#
#
# def numpy_append(...):
#   return np.append(...)
#
#
# def numpy_extract(...):
#   return np.extract(...)
#
#
# def numpy_trim_zeros(...):
#   return np.trim_zeros(...)

# Label is 0 or 1


def logistic_regression(input, label, W, b):
  h1 = np.dot(input, W) + b
  prediction = 0.5 * (np.tanh(h1) + 1)
  label_probabilities = prediction * label + (1 - prediction) * (1 - label)
  loss = -np.sum(np.log(label_probabilities))
  return loss


def det(sqm):
  return np.linalg.det(sqm)


# ================================================
# TFE grads
# ================================================


def tfe_negative(t):
  return tf.negative(t)


def tfe_exp(t):
  return tf.exp(t)


def tfe_log(t):
  return tf.log(t)


def tfe_tanh(t):
  return tf.tanh(t)


def tfe_cosh(t):
  return tf.cosh(t)


def tfe_sinh(t):
  return tf.sinh(t)


def tfe_rsqrt(t):
  return tf.rsqrt(t)


def tfe_expand_dims_before(t):
  return tf.expand_dims(t, 0)


def tfe_expand_dims_after(t):
  return tf.expand_dims(t, 1)


def tfe_squeeze_before(t):
  return tf.squeeze(tf.expand_dims(t, 0), 0)


def tfe_reshape_flat(t):
  return tf.reshape(t, (-1,))


def tfe_reshape_noop(t):
  # TODO: Why doesn't test_forward see that shape is non-differentiable?
  return tf.reshape(tf.reshape(t, (-1,)), tf.shape(t))


def tfe_reduce_sum(timage, boolean):
  return tf.reduce_sum(timage, None, boolean)


def tfe_reduce_sum_axis(timage, boolean):
  return tf.reduce_sum(timage, [0, 1, 2], boolean)


def tfe_reduce_mean(timage, boolean):
  return tf.reduce_mean(timage, None, boolean)


def tfe_reduce_mean_axis(timage, boolean):
  return tf.reduce_mean(timage, [0, 1, 2], boolean)


def tfe_reduce_max(timage, boolean):
  return tf.reduce_max(timage, None, boolean)


def tfe_reduce_max_axis(timage, boolean):
  return tf.reduce_max(timage, [0, 1, 2], boolean)


def tfe_add(t1, t2):
  return tf.add(t1, t2)


def tfe_add_bcast(s, t):
  return tf.add(s, t)


def tfe_subtract(t1, t2):
  return tf.subtract(t1, t2)


def tfe_multiply(t1, t2):
  return tf.multiply(t1, t2)


def tfe_divide(t1, t2):
  return tf.divide(t1, t2)


def tfe_maximum(t1, t2):
  return tf.maximum(t1, t2)


def tfe_squared_difference(t1, t2):
  return tf.squared_difference(t1, t2)


def tfe_matmul(mat1, mat2, boolean1, boolean2):
  return tf.matmul(mat1, mat2, transpose_a=boolean1, transpose_b=boolean2)


def tfe_matmul_highdim(timage1, timage2, boolean1, boolean2):
  return tf.matmul(timage1, timage2, transpose_a=boolean1, transpose_b=boolean2)


def tfe_conv2d(timage, tkernel, conv2dstrides):
  return tf.nn.conv2d(timage, tkernel, conv2dstrides, 'SAME')


def tfe_max_pool(timage, pool2dsizes, conv2dstrides):
  return tf.nn.max_pool(timage, pool2dsizes, conv2dstrides, 'SAME')


def tfe_avg_pool(timage, pool2dsizes, conv2dstrides):
  return tf.nn.avg_pool(timage, pool2dsizes, conv2dstrides, 'SAME')


# ================================================
# Traced TFE calls
# ================================================


@tangent.trace
def _trace_mul(a, b):
  out = tf.multiply(a, b)
  c = 3
  del c  # invalid Tangent syntax, but invisible to tracing
  return out


def tfe_trace_fn(t):
  out = _trace_mul(t, t)
  result = tf.reduce_sum(out)
  return result


def _nontrace_mul(a, b):
  out = tf.multiply(a, b)
  return out


def tfe_notrace_fn(t):
  out = _nontrace_mul(t, t)
  result = tf.reduce_sum(out)
  return result


# ================================================
# Function calls
# ================================================


def _inner_fn(a, b):
  return a * b


def mul_with_call(a, b):
  return _inner_fn(a, b)


def if_mul_with_call(a, b):
  if a > b:
    o = _inner_fn(a, b)
  else:
    o = _inner_fn(b, a)
  return o


def iterpower_with_call(a, n):
  for i in range(n):
    a = _inner_fn(a, a)
  return a


def nested_call(a, b):
  return mul_with_call(a, b)


def iterpower_with_nested_def(a, n):
  def _fn(a, n):
    return a * n

  for i in range(n):
    a = _fn(a, a)
  return a


# ================================================
# Subscripts
# ================================================


def unpacking_args_saxpy(abc_packed_in_tuple):
  a, b, c = abc_packed_in_tuple
  return a * b + c


def dict_saxpy(val):
  return val['a'] * val['b'] + val['c']


def dict_wrapper(abc_packed_in_tuple):
  a, b, c = abc_packed_in_tuple
  return dict_saxpy(dict(a=a, b=b, c=c))


def passthru_pack(a, b, c, i):
  x = a, b, c
  return x[i]


def passthru_unpack(abc_packed_in_tuple):
  a, b, c = abc_packed_in_tuple
  return a

# ================================================
# Misc
# ================================================


def cart2polar(a, b):
  r = np.sqrt(a**2.0 + b**2.0)
  theta = np.arctan(b, a)  # Should really be arctan2
  return r, theta


def inlining_contextmanager(a):
  b = a * a
  with insert_grad_of(a) as g:
    g = g * 0.9
  c = b * a
  return c


def listcomp(a):
  return np.sum([i * 3 for i in a])


def while_big(a):
  while np.abs(a) > 0.1:
    a = a * 0.5
  return a


def active_subscript(x):
  y = np.zeros_like(x)
  for i in range(len(x)):
    y[i] = x[i]
  return np.sum(y)


def init_array_grad_maybe_active(x):
  h = np.zeros((len(x), 3))
  for t in range(len(h)):
    h[t] = x[t]
  return np.sum(h)


def slicing(x):
  return np.sum(x[1:])


def rnn(inputs, W):
  h = np.zeros((4, 3))
  for t in range(3):
    h[t + 1] = np.dot(inputs[t], W) + h[t]
  return np.sum(h[1:])


def augassign(a):
  x = a * a
  x *= x
  return x


def bilinear(x, h, U, W, b):
  y = np.dot(x, W)
  y = y + np.dot(h, U)
  y = y + b
  y = np.tanh(y)
  return np.sum(y)


# Functions were problematic in the development of grad(grad(f))
# They stress differentiation through stack pushes and pops,
# which can complicate dataflow analysis
def stack_pushing(a):
  stack = tangent.Stack()
  y = a * a
  tangent.push(stack, a, 'abc')
  aa = tangent.pop(stack, 'abc')
  z = y * aa
  z = z * aa
  return z


def gradgrad_pow(a, b):
  if a >= 0:
    a = a ** b
  else:
    a = (-a) ** b
  return a


def gradgrad_pow_twice(a):
  a = a * a
  y = a * a
  return y


def gradgrad_chain_of_mults(a):
  b = a  # this line forces overwrite protection
  b = a * a
  c = b * b
  d = c * c
  return d


def gradgrad_chain_of_multcalls(a):
  b = mul(a, a)
  c = mul(b, b)
  d = mul(c, c)
  return d


def useless_stack_ops(a):
  _stack = tangent.Stack()
  b = a * a
  tangent.push(_stack, b, 'abc')
  b = tangent.pop(_stack, 'abc')
  return a


def redefining_var_as_list(a):
  # Initialize the tape
  _stack = tangent.Stack()
  x = a
  tangent.push(_stack, x, 'abc')
  x = [a]
  x = tangent.pop(_stack, 'abc')
  return a


def nested_dict(p):
  return p['i']['j'] * p['i']['k']


def extended_slice(x):
  return np.sum(x[::2, np.newaxis])
