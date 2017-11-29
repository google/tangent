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
"""TensorFlow extensions."""
from __future__ import absolute_import

from numbers import Number

import numpy as np
from tangent import grads
from tangent import non_differentiable
from tangent import tangents
from tangent import utils
from tangent.grads import adjoint
from tangent.tangents import tangent_
from tangent.utils import array_shapes_match
from tangent.utils import register_all_add_grad
from tangent.utils import register_all_shape_checker
from tangent.utils import register_init_grad
from tangent.utils import register_shape_function
from tangent.utils import register_unbroadcast
from tangent.utils import register_unreduce
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops


def size(x, axis):
  axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
  return max(np.prod(axis_shape).value, 1)


def dtype(t):
  return t.dtype


def shape_as_list(t):
  return t.shape.as_list()


def tensor_shapes_match(a, b):
  return tf.shape(a).shape == tf.shape(b).shape


register_shape_function(ops.EagerTensor, shape_as_list)
register_shape_function(resource_variable_ops.ResourceVariable, shape_as_list)


non_differentiable.register_non_differentiable_functions(
    tf.shape, tf.to_float, tf.equal, tf.constant,
    tf.zeros, tf.ones, tf.zeros_like, tf.ones_like,
    size, shape_as_list, dtype)


register_init_grad(ops.EagerTensor, tf.zeros_like)
register_init_grad(resource_variable_ops.ResourceVariable, tf.zeros_like)


register_all_add_grad(
    tf.add, (ops.EagerTensor, resource_variable_ops.ResourceVariable))

register_all_shape_checker(
    tensor_shapes_match,
    (ops.EagerTensor, resource_variable_ops.ResourceVariable))

#
# Utilities
#


def unbroadcast_tfe_to(tensor, shape):
  """Reverse the broadcasting operation.

  See utils.py.

  Args:
    tensor: A Tensor.
    shape: A shape that could have been broadcasted to the shape of tensor.

  Returns:
    Tensor with dimensions summed to match `shape`.
  """
  axis = utils.create_unbroadcast_axis(shape, shape_as_list(tensor))
  return tf.reshape(tf.reduce_sum(tensor, axis=axis), shape)


def unbroadcast_tensor(tensor, like):
  """Reverse the broadcasting operation.

  See utils.py.

  Args:
    tensor: A Tensor.
    like: A Tensor that could have been broadcasted to the shape of tensor.

  Returns:
    Tensor with certain dimensions summed to match the shape of `like`.
  """
  return unbroadcast_tfe_to(tensor, shape_as_list(like))


register_unbroadcast(ops.EagerTensor, unbroadcast_tensor)
register_unbroadcast(resource_variable_ops.ResourceVariable, unbroadcast_tensor)


def unreduce_tensor(tensor, shape, axis, keepdims):
  """Reverse summing over a dimension.

  See utils.py.

  Args:
    tensor: The tensor that was reduced.
    shape: A list, the original shape of the tensor before reduction.
    axis: The axis or axes that were summed.
    keepdims: Whether these axes were kept as singleton axes.

  Returns:
    A tensor with axes broadcast to match the shape of the original tensor.
  """
  if not keepdims:
    if axis is None:
      axis = range(len(shape))
    elif isinstance(axis, int):
      axis = axis,
    for ax in sorted(axis):
      tensor = tf.expand_dims(tensor, ax)
  tile_shape = np.array(shape) / np.array(shape_as_list(tensor))
  return tf.tile(tensor, tile_shape)


register_unreduce(ops.EagerTensor, unreduce_tensor)
register_unreduce(resource_variable_ops.ResourceVariable, unreduce_tensor)


# TODO: Once the optimizer can handle multiple return values, consolidate.
def matmul_adjoint_x(dz, x, y, transpose_a, transpose_b):
  """Implementation of dtfmatmul wrt x, separate for readability."""
  if not transpose_a and not transpose_b:
    return tf.matmul(dz, y, transpose_b=True)
  elif not transpose_a and transpose_b:
    return tf.matmul(dz, y)
  elif transpose_a and not transpose_b:
    return tf.matmul(y, dz, transpose_b=True)
  else:  # transpose_a and transpose_b
    return tf.matmul(y, dz, transpose_a=True, transpose_b=True)


def matmul_adjoint_y(dz, x, y, transpose_a, transpose_b):
  """Implementation of dtfmatmul, separate for readability."""
  if not transpose_a and not transpose_b:
    return tf.matmul(x, dz, transpose_a=True)
  elif not transpose_a and transpose_b:
    return tf.matmul(dz, x, transpose_a=True)
  elif transpose_a and not transpose_b:
    return tf.matmul(x, dz)
  else:  # transpose_a and transpose_b
    return tf.matmul(dz, x, transpose_a=True, transpose_b=True)


#
# Adjoints
#


@adjoint(tf.exp)
def dtfexp(y, x):
  d[x] = y * d[y]


@adjoint(tf.log)
def dtflog(y, x):
  d[x] = d[y] / x


@adjoint(tf.tanh)
def dtftanh(y, x):
  d[x] = d[y] * (1 - (y * y))


@adjoint(tf.cosh)
def dtfcosh(y, x):
  d[x] = d[y] * tf.sinh(x)


@adjoint(tf.sinh)
def dtfsinh(y, x):
  d[x] = d[y] * tf.cosh(x)


@adjoint(tf.rsqrt)
def drsqrt(y, x):
  d[x] = -0.5 * d[y] * tf.pow(tf.conj(y), tf.constant(3.0))


@adjoint(tf.negative)
def dtfnegative(y, x):
  # TODO: Remove the unbroadcast.
  d[x] = tangent.unbroadcast_tensor(tf.negative(d[y]), x)


@adjoint(tf.expand_dims)
def dtfexpand_dims(y, x, axis):
  d[x] = tf.squeeze(d[y], axis)


@adjoint(tf.squeeze)
def dtfsqueeze(y, x, axis=None):
  d[x] = tf.expand_dims(d[y], axis)


@adjoint(tf.reshape)
def dtfreshape(y, x, shape):
  d[x] = tf.reshape(d[y], tf.shape(x))


@adjoint(tf.reduce_sum)
def dtfreduce_sum(y, x, axis=None, keep_dims=False):
  # TODO: We may be able to assume unreduce_tensor works throughout.
  d[x] = tangent.unreduce(d[y], tangent.shape_as_list(x), axis, keep_dims)


@adjoint(tf.reduce_mean)
def dtfreduce_mean(y, x, axis=None, keep_dims=False):
  n = tf.constant(float(tangent.size(x, axis)))
  d[x] = tf.divide(
      tangent.unreduce(d[y], tangent.shape_as_list(x), axis, keep_dims), n)


@adjoint(tf.reduce_max)
def dtfreduce_max(y, x, axis=None, keep_dims=False):
  mask = tf.to_float(
      tf.equal(
          tangent.unreduce(y, tangent.shape_as_list(x), axis, keep_dims), x))
  d[x] = tf.multiply(
      tangent.unreduce(d[y], tangent.shape_as_list(x), axis, keep_dims), mask)


@adjoint(tf.add)
def dtfadd(z, x, y):
  d[x] = tangent.unbroadcast(d[z], x)
  d[y] = tangent.unbroadcast(d[z], y)


@adjoint(tf.subtract)
def dtfsubtract(z, x, y):
  d[x] = tangent.unbroadcast(d[z], x)
  d[y] = tangent.unbroadcast(tf.negative(d[z]), y)


@adjoint(tf.multiply)
def dtfmultiply(z, x, y):
  d[x] = tangent.unbroadcast(tf.multiply(d[z], y), x)
  d[y] = tangent.unbroadcast(tf.multiply(d[z], x), y)


@adjoint(tf.divide)
def dtfdivide(z, x, y):
  d[x] = tangent.unbroadcast(tf.divide(d[z], y), x)
  d[y] = tangent.unbroadcast(
      tf.negative(tf.divide(tf.multiply(d[z], x), tf.multiply(y, y))), y)


@adjoint(tf.maximum)
def dtfmaximum(z, x, y):
  d[x] = tf.multiply(d[z], tf.to_float(tf.equal(z, x)))
  d[y] = tf.multiply(d[z], tf.to_float(tf.equal(z, y)))


@adjoint(tf.squared_difference)
def dtfsquared_difference(z, x, y):
  d[x] = tangent.unbroadcast(2 * d[z] * (x - y), x)
  d[y] = tangent.unbroadcast(2 * d[z] * (y - x), y)


@adjoint(tf.matmul)
def dtfmatmul(z, x, y, transpose_a=False, transpose_b=False):
  d[x] = tangent.matmul_adjoint_x(d[z], x, y, transpose_a, transpose_b)
  d[y] = tangent.matmul_adjoint_y(d[z], x, y, transpose_a, transpose_b)


@adjoint(tf.nn.conv2d)
def dtfconv2d(z, x, y, strides, padding):
  d[x] = tf.nn.conv2d_backprop_input(tf.shape(x), y, d[z], strides, padding)
  d[y] = tf.nn.conv2d_backprop_filter(x, tf.shape(y), d[z], strides, padding)


@adjoint(tf.nn.conv2d_backprop_input)
def dtfconv2d_backprop_input(z, shape, x, y, strides, padding):
  # TODO: Add tests.
  d[x] = tf.nn.conv2d_backprop_filter(d[z], shape, y, strides, padding)
  d[y] = tf.nn.conv2d(d[z], x, strides, padding)


@adjoint(tf.nn.conv2d_backprop_filter)
def dtfconv2d_backprop_filter(z, x, shape, y, strides, padding):
  # TODO: Add tests.
  d[x] = tf.nn.conv2d_backprop_input(shape, d[z], y, strides, padding)
  d[y] = tf.nn.conv2d(x, d[z], strides, padding)


@adjoint(tf.nn.avg_pool)
def dtfavg_pool(y, x, sizes, strides, padding):
  # TODO: We shouldn't rely on private modules.
  d[x] = tf.nn._nn_grad.gen_nn_ops._avg_pool_grad(
      tf.shape(x), d[y], sizes, strides, padding)


@adjoint(tf.nn.max_pool)
def dtfmax_pool(y, x, sizes, strides, padding):
  # TODO: We shouldn't rely on private modules.
  d[x] = tf.nn._nn_grad.gen_nn_ops._max_pool_grad(
      x, y, d[y], sizes, strides, padding)


#
# Tangents
#


@tangent_(shape_as_list)
def tshape_as_list(y, x):
  d[y] = tangent.shape_as_list(d[x])


@tangent_(tf.exp)
def ttfexp(y, x):
  d[y] = d[x] * y


@tangent_(tf.log)
def ttflog(y, x):
  d[y] = d[x] / x


@tangent_(tf.tanh)
def ttftanh(y, x):
  cx = tf.cosh(x)
  d[y] = d[x] / (cx * cx)


@tangent_(tf.cosh)
def ttfcosh(y, x):
  d[y] = d[x] * tf.sinh(x)


@tangent_(tf.sinh)
def ttfsinh(y, x):
  d[y] = d[x] * tf.cosh(x)


@tangent_(tf.expand_dims)
def ttfexpand_dims(y, x, axis):
  d[y] = tf.expand_dims(d[x], axis)


@tangent_(tf.squeeze)
def ttfsqueeze(y, x, axis):
  d[y] = tf.squeeze(d[x], axis)


@tangent_(tf.reshape)
def ttfreshape(y, x, shape):
  d[y] = tf.reshape(d[x], shape)


@tangent_(tf.reduce_sum)
def ttfreduce_sum(y, x, axis=None, keep_dims=False):
  d[y] = tf.reduce_sum(d[x], axis, keep_dims)


@tangent_(tf.reduce_mean)
def ttfreduce_mean(y, x, axis=None, keep_dims=False):
  d[y] = tf.reduce_mean(d[x], axis, keep_dims)


@tangent_(tf.reduce_max)
def ttfreduce_max(y, x, axis=None, keep_dims=False):
  mask = tf.to_float(
      tf.equal(
          tangent.unreduce(
              tf.ones_like(y), tangent.shape_as_list(x), axis, keep_dims), x))
  d[y] = tf.multiply(d[x], mask)


@tangent_(tf.negative)
def ttfnegative(y, x):
  d[y] = tf.negative(d[x])


@tangent_(tf.add)
def ttfadd(z, x, y):
  d[z] = tf.add(d[x], d[y])


@tangent_(tf.subtract)
def ttfsubtract(z, x, y):
  d[z] = tf.subtract(d[x], d[y])


@tangent_(tf.multiply)
def ttfmultiply(z, x, y):
  d[z] = tf.add(tf.multiply(d[x], y), tf.multiply(x, d[y]))


@tangent_(tf.divide)
def ttfdivide(z, x, y):
  d[z] = tf.divide(
          tf.subtract(tf.multiply(d[x], y), tf.multiply(x, d[y])),
          tf.multiply(y, y))


@tangent_(tf.maximum)
def ttfmaximum(z, x, y):
  d[z] = d[x] * tf.to_float(tf.equal(z, x)) + d[y] * tf.to_float(tf.equal(z, y))


@tangent_(tf.nn.avg_pool)
def ttfavg_pool(y, x, sizes, strides, padding):
  raise tangent.ForwardNotImplementedError(tf.nn.avg_pool)


@tangent_(tf.nn.max_pool)
def ttfmax_pool(y, x, sizes, strides, padding):
  raise tangent.ForwardNotImplementedError(tf.nn.max_pool)


@tangent_(tf.shape)
def tshape(y, x):
  d[y] = tf.shape(d[x])


#
# Blacklist unimplemented Eager grads
#

grads.UNIMPLEMENTED_ADJOINTS.update(
    grads.get_module_functions((tf, tf.distributions, tf.image, tf.layers,
                                tf.linalg, tf.losses,
                                tf.nn)) - set(grads.adjoints))

tangents.UNIMPLEMENTED_TANGENTS.update(
    grads.get_module_functions((tf, tf.distributions, tf.image, tf.layers,
                                tf.linalg, tf.losses,
                                tf.nn)) - set(tangents.tangents))
