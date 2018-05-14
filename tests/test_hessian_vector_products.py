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
"""Tests for Hessian-vector products on a few limited functions.

HVPs are run in three configurations:
 - Reverse-over-reverse (autograd-style)
 - Forward-over-reverse (traditional AD style, most efficient)
 - Reverse-over-forward
"""

from autograd import hessian_vector_product
import autograd.numpy as np
import pytest
import tangent
import tfe_utils

@pytest.fixture
def tf():
  try:
    import tensorflow as tf
  except ImportError:
    pytest.skip()


# This test function broke HVPs several times
# during development, so we're using it as a unit test.
def f_straightline(x):
  a = x * x * x
  b = a * x**2.0
  return np.sum(b)


def cube(a):
  b = a * a * a
  return b


def f_calltree(x):
  a = cube(x)
  b = a * x**2.0
  return np.sum(b)


def tf_straightline(x, tf):
  a = x * x * x
  b = a * x ** 2.0
  return tf.reduce_sum(b)


def _test_hvp(func, optimized):
  np.random.seed(0)
  a = np.random.normal(scale=1, size=(300,)).astype('float32')
  v = a.ravel()

  modes = ['forward', 'reverse']
  for mode1 in modes:
    for mode2 in modes:
      if mode1 == mode2 == 'forward':
        continue
      df = tangent.autodiff(
          func,
          mode=mode1,
          motion='joint',
          optimized=optimized,
          check_dims=False)
      ddf = tangent.autodiff(
          df, mode=mode2, motion='joint', optimized=optimized, check_dims=False)
      dx = ddf(a, 1, v)
      hvp_ag = hessian_vector_product(func)
      dx_ag = hvp_ag(a, v)
      assert np.allclose(dx, dx_ag)


def _test_tf_hvp(func, optimized, tf):
  a = tf.random_normal(shape=(300,))
  v = tf.reshape(a, shape=(-1,))

  modes = ['forward', 'reverse']
  for mode1 in modes:
    for mode2 in modes:
      if mode1 == mode2 == 'forward':
        continue
      df = tangent.autodiff(
          func,
          mode=mode1,
          motion='joint',
          optimized=optimized,
          check_dims=False)
      ddf = tangent.autodiff(
          df, mode=mode2, motion='joint', optimized=optimized, check_dims=False)
      dx = ddf(a, tf.constant(1.0), v)
      # We just ensure it computes something in this case.
      assert dx.shape == a.shape


def test_hvp_complex_tf(optimized, tf):
  _test_tf_hvp(tf_straightline, optimized, tf)


def test_hvp_straightline(optimized, tf):
  _test_hvp(f_straightline, optimized, tf)


def test_hvp_calltree(optimized, tf):
  _test_hvp(f_calltree, optimized, tf)


if __name__ == '__main__':
  assert not pytest.main([__file__, '--short'])
