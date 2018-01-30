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
"""Tests for the demos."""
import sys
import numpy as np
import pytest
from tangent.demos import models
from tangent.demos import train_utils
from tangent.demos.common_kernels import avgpool2d
from tangent.demos.common_kernels import batch_norm
from tangent.demos.common_kernels import batch_normalization
from tangent.demos.common_kernels import conv2d
from tangent.demos.common_kernels import dense
from tangent.demos.common_kernels import logsoftmax
from tangent.demos.common_kernels import logsumexp
from tangent.demos.common_kernels import maxpool2d
from tangent.demos.common_kernels import moments
from tangent.demos.common_kernels import moving_average
from tangent.demos.common_kernels import relu
from tangent.demos.common_kernels import softmax
from tangent.demos.common_kernels import softmax_crossent
from tangent.demos.common_kernels import softmax_crossent_numpy
from tangent.grad_util import grad
import tfe_utils
import utils

import tensorflow as tf


def _check_grad_computes(f, inputs, params, state, hparams, lr):
  # We can't verify convergence - it's just too flaky,
  df = grad(f, (1,), True)
  y = f(inputs, params, state, hparams)
  dp = df(inputs, params, state, hparams, tf.ones_like(y))
  newp = train_utils.map_dicts(lambda v, dv: v - lr * dv, params, dp)
  f(inputs, newp, state, hparams)


def _test_image():
  return tf.random_normal((3, 5, 7, 11))


def _test_kernel():
  return tf.random_normal((3, 3, 11, 2))


def test_moments_equivalent_with_tf():
  x = _test_image()
  our_moms = tfe_utils.tensors_to_numpy(moments(x, (0, 1, 2)))
  tf_moms = tfe_utils.tensors_to_numpy(tf.nn.moments(x, (0, 1, 2)))
  for ours, tfs in zip(our_moms, tf_moms):
    assert np.allclose(ours, tfs), ('Expected: %s\nGot: %s' % (ours, tfs))


def test_moments(motion, optimized):
  # TFE can only handle single return values.
  def moments_1(x, axis):
    mean, var = moments(x, axis)
    return mean
  def moments_2(x, axis):
    mean, var = moments(x, axis)
    return var
  x = _test_image()
  tfe_utils.test_rev_tensor(moments_1, motion, optimized, False, (0,), x, [0])
  tfe_utils.test_rev_tensor(moments_2, motion, optimized, False, (0,), x, [0])
  tfe_utils.test_rev_tensor(moments_1, motion, optimized, False, (0,), x, [1])
  tfe_utils.test_rev_tensor(moments_2, motion, optimized, False, (0,), x, [1])
  tfe_utils.test_rev_tensor(moments_1, motion, optimized, False, (0,), x,
                            [0, 1])
  tfe_utils.test_rev_tensor(moments_2, motion, optimized, False, (0,), x,
                            [0, 1])
  tfe_utils.test_rev_tensor(moments_1, motion, optimized, False, (0,), x,
                            None)
  tfe_utils.test_rev_tensor(moments_2, motion, optimized, False, (0,), x,
                            None)


def test_moving_average(motion, optimized):
  x = tf.constant(1.0)
  avg = tf.constant(0.0)
  mom = tf.constant(0.5)

  tfe_utils.test_rev_tensor(moving_average, motion, optimized, False, (0,), x,
                            avg, mom)


def test_batch_normalization(motion, optimized):
  x = tf.constant([[[1.0, 4.0]]])
  mean, variance = moments(x, (0, 1, 2))
  offset = tf.constant(0.5)
  scale = tf.constant(0.5)
  args = (x, mean, variance, offset, scale)
  tfe_utils.test_rev_tensor(batch_normalization, motion, optimized, False,
                            (0, 3, 4), *args)


def test_batch_norm(motion, optimized):
  x = tf.constant([[[1.0, 4.0]]])
  axis = (0, 1, 2)
  moving_mean, moving_var = moments(x, axis)
  momentum = tf.constant(0.9)
  offset = tf.constant(0.5)
  scale = tf.constant(0.5)
  args = (x, moving_mean, moving_var, momentum, offset, scale, axis, True)
  tfe_utils.test_rev_tensor(batch_norm, motion, optimized, False, (0, 4, 5),
                            *args)

  args = (x, moving_mean, moving_var, momentum, offset, scale, axis, False)
  tfe_utils.test_rev_tensor(batch_norm, motion, optimized, False, (0, 4, 5),
                            *args)


def test_conv2d(motion, optimized):
  x = _test_image()
  ker = _test_kernel()
  strides = None
  padding = 'SAME'
  args = (x, ker, strides, padding)
  tfe_utils.test_rev_tensor(conv2d, motion, optimized, False, (0, 1), *args)


def test_relu(motion, optimized):
  x = _test_image()
  tfe_utils.test_rev_tensor(relu, motion, optimized, False, (0,), x)


def test_maxpool2d(motion, optimized):
  x = _test_image()
  sizes = (2, 2)
  strides = None
  padding = 'SAME'
  args = (x, sizes, strides, padding)
  tfe_utils.test_rev_tensor(maxpool2d, motion, optimized, False, (0,), *args)


def test_avgpool2d(motion, optimized):
  x = _test_image()
  sizes = (2, 2)
  strides = None
  padding = 'SAME'
  args = (x, sizes, strides, padding)
  tfe_utils.test_rev_tensor(avgpool2d, motion, optimized, False, (0,), *args)


def test_dense(motion, optimized):
  x = tf.random_normal((3, 4))
  w = tf.random_normal((4, 5))
  b = tf.random_normal((5,))
  args = (x, w, b)
  tfe_utils.test_rev_tensor(dense, motion, optimized, False, (0, 1, 2), *args)

  y = dense(x, w, b)
  assert y.shape == (3, 5)


def test_softmax(motion, optimized):
  del motion
  del optimized
  x = tf.random_normal((3, 5))
  # TODO: Test the derivative once we have tensor slicing support.

  y = softmax(x)
  assert y.shape == (3, 5)


def test_logsumexp(motion, optimized):
  x = tf.random_normal((3, 4))
  tfe_utils.test_rev_tensor(logsumexp, motion, optimized, False, (0,), x,
                            [-1], False)


def test_logsoftmax(motion, optimized):
  del motion
  del optimized
  x = tf.random_normal((3, 5))
  # TODO: Test the derivative once we have tensor slicing support.

  y = logsoftmax(x)
  assert y.shape == (3, 5)


def test_softmax_crossent(motion, optimized):
  def reduced_crossent(logits, y):
    return tf.reduce_mean(softmax_crossent(logits, y))

  y = tf.to_float(tf.random_uniform((3, 5), 0, 2, dtype=tf.int32))
  logits = tf.random_normal((3, 5), dtype=tf.float32)
  tfe_utils.test_rev_tensor(reduced_crossent, motion, optimized, False, (0,),
                            logits, y)

  y = reduced_crossent(logits, y)
  assert y.numpy() >= 0


def test_softmax_crossent_numpy(motion, optimized):
  def reduced_crossent(logits, y):
    return np.mean(softmax_crossent_numpy(logits, y))

  y = np.random.uniform(0, 2, size=(3, 5))
  logits = np.random.normal(size=(3, 5))
  utils.test_reverse_array(
      reduced_crossent, motion, optimized, False, logits, y)

  y = reduced_crossent(logits, y)
  assert y >= 0


def test_dense_layer():
  x = tf.random_normal((10, 10))
  params, state, hparams = models.dense_layer_params(
      input_shape=(10, 10),
      num_units=1,
      activation='relu')
  _check_grad_computes(models.dense_layer, x, params, state, hparams, 1e-3)


def test_pooling_layer():
  x = tf.random_normal((1, 4, 4, 3))
  params, state, hparams = models.pooling_layer_params(
      input_shape=(1, 4, 4, 3),
      pooling='max',
      size=2,
      strides=2)
  _check_grad_computes(models.pooling_layer, x, params, state, hparams, 1e-3)


def test_resnet_conv_layer():
  x = tf.random_normal((1, 4, 4, 3))
  params, state, hparams = models.resnet_conv_layer_params(
      input_shape=(1, 4, 4, 3),
      num_filters=1,
      kernel_size=3,
      strides=1,
      bn_momentum=0.99,
      activation='relu')
  _check_grad_computes(models.resnet_conv_layer, x, params, state, hparams, 1e-11)


def test_resnet_conv_block():
  x = tf.random_normal((1, 4, 4, 3))
  params, state, hparams = models.resnet_conv_block_params(
      input_shape=(1, 4, 4, 3),
      num_filters=[1, 1, 1],
      kernel_size=3,
      strides=2,
      bn_momentum=0.99,
      activation='relu')
  _check_grad_computes(models.resnet_conv_block, x, params, state, hparams, 1e-13)


def test_resnet_identity_block():
  x = tf.random_normal((1, 4, 4, 3))
  params, state, hparams = models.resnet_identity_block_params(
      input_shape=(1, 4, 4, 3),
      num_filters=[1, 1, 1],
      kernel_size=3,
      bn_momentum=0.99,
      activation='relu')
  _check_grad_computes(
      models.resnet_identity_block, x, params, state, hparams, 1e-13)


def test_resnet_50():
  x = tf.random_normal((10, 28, 28, 1))
  params, state, hparams = models.resnet_50_params(
      input_shape=(10, 28, 28, 1), classes=2, bn_momentum=0.99)
  assert models.resnet_50(x, params, state, hparams).shape == (10, 2)
  _check_grad_computes(models.resnet_50, x, params, state, hparams, 1e-13)


def test_sgd():
  p = tf.get_variable('test_sgd', shape=(3, 3, 3, 3))
  dp = tf.random_normal((3, 3, 3, 3))

  opt = train_utils.sgd()
  upd = opt(p, dp, 0.1)  # p remains unchanged

  opt = tf.train.GradientDescentOptimizer(0.1)
  opt.apply_gradients([(dp, p)])  # Results will now be in p

  assert np.allclose(p.read_value().numpy(), upd.numpy())


def test_momentum():
  p = tf.get_variable('test_momentum', shape=(3, 3, 3, 3))
  dp = tf.random_normal((3, 3, 3, 3))

  opt = train_utils.momentum(0.98, p)
  # Apply twice to check the momentum dynamics.
  upd = opt(opt(p, dp, 0.1), dp, 0.1)  # p remains unchanged

  opt = tf.train.MomentumOptimizer(0.1, 0.98)
  opt.apply_gradients([(dp, p)])
  opt.apply_gradients([(dp, p)])  # Results will now be in p

  assert np.allclose(p.read_value().numpy(), upd.numpy())


def test_adam():
  p = tf.get_variable('test_adam', shape=(2,))
  dp = tf.random_normal((2,))

  opt = train_utils.adam(0.9, 0.999, 1e-8, p)
  # Apply twice to check the momentum dynamics.
  upd = opt(opt(p, dp, 0.1), dp, 0.1)  # p remains unchanged

  opt = tf.train.AdamOptimizer(0.1, 0.9, 0.999, 1e-8)
  opt.apply_gradients([(dp, p)])
  opt.apply_gradients([(dp, p)])  # Results will now be in p

  tf_val = p.read_value().numpy()
  our_val = upd.numpy()
  assert np.allclose(tf_val, our_val, atol=1e-5)

if __name__ == '__main__':
  assert not pytest.main([__file__, '--short'] + sys.argv[1:])
