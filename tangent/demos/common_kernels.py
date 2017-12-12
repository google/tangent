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
"""Common kernels used by demos."""
from __future__ import absolute_import

import os

import autograd.numpy as np

if os.environ.get('USES_TORCH', False):
  import torch
else:
  from tangent import dtype
  import tensorflow as tf


def moments(x, axis):
  mean = tf.reduce_mean(x, axis, keep_dims=True)
  variance = tf.reduce_mean(
      tf.squared_difference(x, mean), axis, keep_dims=True)
  return mean, variance


def moving_average(x, prev_average, momentum):
  moving_mean = tf.add(tf.multiply(tf.subtract(prev_average, x), momentum), x)
  return moving_mean


def batch_normalization(x, mean, variance, offset, scale):
  epsilon = tf.constant(1e-7)
  invv = tf.multiply(tf.rsqrt(tf.add(variance, epsilon)), scale)
  result = tf.add(tf.multiply(tf.subtract(x, mean), invv), offset)
  return result


def batch_norm(x,
               moving_mean,
               moving_variance,
               momentum,
               offset,
               scale,
               axis,
               update=False):
  if update:
    mean, variance = moments(x, axis)
  else:
    mean, variance = moving_mean, moving_variance
  x = batch_normalization(x, mean, variance, offset, scale)
  if update:
    moving_mean = moving_average(moving_mean, mean, momentum)
    moving_variance = moving_average(moving_variance, variance, momentum)
  return x, moving_mean, moving_variance


def conv2d(x, kernel, strides=None, padding='SAME'):
  if strides is None:
    strides = (1, 1, 1, 1)
  return tf.nn.conv2d(x, kernel, strides, padding)


def relu(x):
  return tf.maximum(x, tf.zeros((), dtype=dtype(x)))


def maxpool2d(x, sizes, strides=None, padding='SAME'):
  if strides is None:
    strides = (1, 1, 1, 1)
  sizes = (1,) + sizes + (1,)
  return tf.nn.max_pool(x, sizes, strides, padding)


def avgpool2d(x, sizes, strides=None, padding='SAME'):
  if strides is None:
    strides = (1, 1, 1, 1)
  sizes = (1,) + sizes + (1,)
  return tf.nn.avg_pool(x, sizes, strides, padding)


def dense(x, w, b):
  return tf.add(tf.matmul(x, w), b)


def sigmoid(logits):
  return 1.0 / (1.0 + tf.exp(-logits))


def softmax(logits):
  exp_logits = tf.exp(logits)
  sum_exp_logits = tf.reduce_sum(exp_logits, axis=[-1], keep_dims=True)
  return exp_logits / (1e-8 + sum_exp_logits)


def logsumexp(x, axis=None, keep_dims=False):
  return tf.log(1e-8 + tf.reduce_sum(tf.exp(x), axis=axis, keep_dims=keep_dims))


def logsoftmax(logits):
  return tf.subtract(logits, logsumexp(logits, axis=[-1], keep_dims=True))


def softmax_crossent(logits, y):
  return tf.negative(tf.reduce_sum(
      tf.multiply(logsoftmax(logits), y), axis=[-1]))


def logsumexp_numpy(x, axis=None, keep_dims=False):
  return np.log(1e-8 + np.sum(np.exp(x), axis=axis, keepdims=keep_dims))


def logsoftmax_numpy(logits):
  return logits - logsumexp_numpy(logits, axis=-1, keep_dims=True)


def softmax_crossent_numpy(logits, y):
  return -np.sum(logsoftmax_numpy(logits) * y, axis=-1)


def logsumexp_pytorch(x, axis=None, keep_dims=False):
  return torch.log(1e-8 + torch.sum(torch.exp(x), dim=axis, keepdim=keep_dims))


def logsoftmax_pytorch(logits):
  return logits - logsumexp_pytorch(logits, axis=-1, keep_dims=True)


def softmax_crossent_pytorch(logits, y):
  return -torch.sum(
      logsoftmax_pytorch(logits) * y.type(torch.FloatTensor), dim=-1)
