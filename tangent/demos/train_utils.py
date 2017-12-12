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

if not os.environ.get('USES_TORCH', False):
  import tensorflow as tf


def broadcast_update(d, **overrides):
  for k in d:
    if isinstance(d[k], dict):
      broadcast_update(d[k], **overrides)
    elif k in overrides:
      d[k] = overrides[k]


def map_dicts(fn, *dicts):
  if isinstance(dicts[0], dict):
    assert all(
        frozenset(d1.keys()) == frozenset(d2.keys())
        for d2 in dicts for d1 in dicts)
    return {
        k: map_dicts(fn, *tuple(d[k] for d in dicts))
        for k in dicts[0]}
  return fn(*dicts)


def copy_dict(d):
  return map_dicts(lambda x: x, d)


def flatten_dict(d, l):
  if isinstance(d, dict):
    sorted_keys = sorted(d.keys())
    for k in sorted_keys:
      flatten_dict(d[k], l)
  else:
    l.append(d)


def sgd():
  def updates(params, dparams, lr):
    return map_dicts(lambda v, dv: v - lr * dv, params, dparams)
  return updates


def momentum(beta, params):
  m = map_dicts(tf.zeros_like, params)
  momenta = [m]

  def updates(params, dparams, lr):
    m, = momenta
    m = map_dicts(lambda m, dp: beta * m + dp, m, dparams)
    momenta[0] = m

    return map_dicts(lambda m, p: p - lr * m, m, params)

  return updates


def adam(beta1, beta2, eps, params):
  m = map_dicts(tf.zeros_like, params)
  v = map_dicts(tf.zeros_like, params)
  momenta = [m, v]
  powers = [beta1, beta2]

  def updates(params, dparams, lr):
    lr_t = lr * tf.sqrt(1 - powers[1]) / (1 - powers[0])
    m, v = momenta
    m = map_dicts(lambda m, dp: beta1 * m + (1 - beta1) * dp, m, dparams)
    v = map_dicts(lambda v, dp: beta2 * v + (1 - beta2) * dp * dp, v, dparams)

    powers[0] *= beta1
    powers[1] *= beta2
    momenta[0] = m
    momenta[1] = v

    return map_dicts(
        lambda m, v, p: p - lr_t * m / (tf.sqrt(v) + eps), m, v, params)

  return updates
