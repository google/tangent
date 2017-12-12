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
"""Simple model builders that have a very small number of parameters."""
from __future__ import absolute_import

import os

import autograd.numpy as np
from tangent.demos.common_kernels import softmax_crossent
from tangent.demos.common_kernels import softmax_crossent_numpy
from tangent.demos.common_kernels import softmax_crossent_pytorch

if os.environ.get('USES_TORCH', False):
  import torch
else:
  import tensorflow as tf


def mlp_numpy(x, w1, b1, wout, bout, label):
  """Basic MLP demo, implemented in NumPy."""
  h1 = np.tanh(np.dot(x, w1) + b1)
  out = np.dot(h1, wout) + bout
  loss = np.mean(softmax_crossent_numpy(out, label))
  return loss


def mlp_tf(x, w1, b1, wout, bout, label):
  """Basic MLP demo, implemented in TensorFlow. Equivalent with `mlp_numpy`."""
  h1 = tf.tanh(tf.add(tf.matmul(x, w1), b1))
  out = tf.add(tf.matmul(h1, wout), bout)
  loss = tf.reduce_mean(softmax_crossent(out, label))
  return loss


def mlp_pytorch(x, w1, b1, wout, bout, label):
  """Basic MLP demo, implemented in PyTorch. Equivalent with `mlp_numpy`."""
  h1 = torch.tanh(x.mm(w1) + b1)
  out = h1.mm(wout) + bout
  # PyTorch does have its own implementation of cross entropy
  # torch.nn.CrossEntropyLoss(), but we want a consistent implementation here.
  loss = torch.mean(softmax_crossent_pytorch(out, label))
  return loss


def rnn_numpy(x, w1, b1, wout, bout, label, num_steps):
  """Basic RNN demo, implemented in NumPy."""
  h1 = x
  for _ in range(num_steps):
    h1 = np.tanh(np.dot(h1, w1) + b1)
  out = np.dot(h1, wout) + bout
  loss = np.mean(softmax_crossent_numpy(out, label))
  return loss


def rnn_tf(x, w1, b1, wout, bout, label, num_steps):
  """Basic RNN demo, implemented in TensorFlow. Equivalent with `rnn_numpy`."""
  h1 = x
  for _ in range(num_steps):
    h1 = tf.tanh(tf.add(tf.matmul(h1, w1), b1))
  out = tf.add(tf.matmul(h1, wout), bout)
  loss = tf.reduce_mean(softmax_crossent(out, label))
  return loss


def simple_loop_numpy(x, num_steps):
  for _ in range(num_steps):
    if np.sum(x) > 1:
      x /= 2
  return np.sum(x)


def simple_loop_tf(x, num_steps):
  for _ in range(num_steps):
    if tf.reduce_sum(x) > 1:
      x /= 2
  return tf.reduce_sum(x)


def simple_loop_pytorch(x, num_steps):
  for _ in range(num_steps):
    # TODO: Is this really the best way to do it?
    if (torch.sum(x) > 1).data[0]:
      x /= 2
  return torch.sum(x)


def simple_loop_tf_pure(x, num_steps):
  """The TensorFlow equivalent of `simple_loop_numpy`."""
  def cond(i, _):
    return i < num_steps

  def body(i, x):
    result = tf.cond(tf.reduce_sum(x) > 1, lambda: x / 2, lambda: x)
    return tf.add(i, 1), result

  i = tf.constant(0)
  _, ret_x = tf.while_loop(cond, body, [i, x])
  return tf.reduce_sum(ret_x)
