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
"""TFE-specific test utils."""
import numpy as np

import pytest
from tangent.grad_util import autodiff, jvp
import utils

try:
  import tensorflow as tf
  from tensorflow.contrib.eager.python import tfe
except ImportError:
  tf = None
  tfe = None
else:
  tfe.enable_eager_execution()


def register_parametrizations(metafunc, short):
  """Create additional parametrizations required for TF tests."""

  for arg in ['t', 't1', 't2']:
    # Note: care must be exercised when sharing tensor objects. Although
    # immutable, references to the same value will be interpreted as the same
    # variable, with unexpected side effects.
    if tf:
      vectors = [
          np.random.randn(i)
          for i in (
              (3,) if short else (3, 5, 10))]
      tensors = [tf.constant(v, dtype=tf.float32) for v in vectors]
    else:
      tensors = [pytest.mark.skip(None, reason='tensorflow not present')(None)]
    if arg in metafunc.fixturenames:
      metafunc.parametrize(arg, tensors)

  for arg in ['mat1', 'mat2']:
    if tf:
      matrices = [
        np.random.randn(*i)
        for i in (
            ((3, 3),) if short else (
                (1, 1),
                (3, 3),
                (5, 5)))]
      tensors = [tf.constant(m, dtype=tf.float32) for m in matrices]
    else:
      tensors = [pytest.mark.skip(None, reason='tensorflow not present')(None)]
    if arg in metafunc.fixturenames:
      metafunc.parametrize(arg, tensors)

  if 's' in metafunc.fixturenames:
    if tf:
      if short:
        scalars = [tf.constant(1.0)]
      else:
        scalars = [tf.constant(c) for c in (0.0, 1.0, 2.0)]
    else:
      scalars = [pytest.mark.skip(reason='tensorflow not present')(None)]
    metafunc.parametrize('s', scalars)

  for arg in ['timage', 'timage1', 'timage2']:
    if arg in metafunc.fixturenames:
      if tf:
        images = [
            np.random.randn(*i)
            for i in (
              ((2, 3, 3, 3),) if short else (
                    (2, 1, 1, 3),
                    (2, 3, 3, 3),
                    (2, 5, 5, 3),
                ))
        ]
        timages = [tf.constant(v, dtype=tf.float32) for v in images]
      else:
        timages = [pytest.mark.skip(reason='tensorflow not present')(None)]
      metafunc.parametrize(arg, timages)

  if 'tkernel' in metafunc.fixturenames:
    if tf:
      kernels = [
          np.random.randn(*i)
          for i in (
              ((3, 3, 3, 1),) if short else (
                  (3, 3, 3, 1),
                  (3, 3, 3, 2),
                  (5, 5, 3, 3),
              ))
      ]
      tkernels = [tf.constant(v, dtype=tf.float32) for v in kernels]
    else:
      tkernels = [pytest.mark.skip(reason='tensorflow not present')(None)]
    metafunc.parametrize('tkernel', tkernels)

  if 'conv2dstrides' in metafunc.fixturenames:
    strides = [(1, 2, 2, 1)] if short else [
        (1, 1, 1, 1),
        (1, 2, 2, 1),
        (1, 2, 2, 2),
    ]
    metafunc.parametrize('conv2dstrides', strides)

  if 'pool2dsizes' in metafunc.fixturenames:
    sizes = [(1, 2, 2, 1)] if short else [
        (1, 1, 1, 1),
        (1, 2, 2, 1),
        (1, 3, 3, 1),
    ]
    metafunc.parametrize('pool2dsizes', sizes)


def tensors_to_numpy(tensors):
  if isinstance(tensors, (tuple, list)):
    return tuple(tensors_to_numpy(t) for t in tensors)
  if isinstance(tensors, tf.Tensor):
    return tensors.numpy()
  raise ValueError('Don\'t know how to handle %s' % type(tensors))


def as_numpy_sig(func):
  """Wrap a TF Eager function into a signature that uses NumPy arrays."""
  def wrapped(*args):
    np_args = [tf.constant(a) if isinstance(a, np.ndarray) else a for a in args]
    return tensors_to_numpy(func(*np_args))
  return wrapped


def test_forward_tensor(func, wrt, *args):
  """Test gradients of functions with TFE signatures."""

  def tangent_func():
    df = jvp(func, wrt=wrt, optimized=True, verbose=True)
    args_ = args + tuple(tf.ones_like(args[i]) for i in wrt)  # seed gradient
    return tensors_to_numpy(df(*args_))

  def reference_func():
    return tensors_to_numpy(tfe.gradients_function(func, params=wrt)(*args))

  def backup_reference_func():
    func_ = as_numpy_sig(func)
    args_ = tensors_to_numpy(args)
    return utils.numeric_grad(utils.numeric_grad(func_))(*args_)

  # TODO: Should results really be that far off?
  utils.assert_result_matches_reference(
      tangent_func, reference_func, backup_reference_func,
      tolerance=1e-4)


def test_gradgrad_tensor(func, optimized, *args):
  """Test gradients of functions with TFE signatures."""

  def tangent_func():
    df = tangent.autodiff(func, motion='joint', optimized=optimized, verbose=True)
    ddf = tangent.autodiff(df, motion='joint', optimized=optimized, verbose=True)
    dxx = ddf(*args)
    return tuple(t.numpy() for t in dxx)

  def reference_func():
    dxx = tfe.gradients_function(tfe.gradients_function(func))(*args)
    return tensors_to_numpy(tuple(t.numpy() for t in dxx))

  def backup_reference_func():
    func_ = as_numpy_sig(func)
    args_ = tensors_to_numpy(args)
    return utils.numeric_grad(utils.numeric_grad(func_))(*args_)

  utils.assert_result_matches_reference(
      tangent_func, reference_func, backup_reference_func,
      tolerance=1e-2)  # extra loose bounds for 2nd order grad


def test_rev_tensor(func, motion, optimized, preserve_result, wrt, *args):
  """Test gradients of functions with TFE signatures."""

  def tangent_func():
    y = func(*args)
    if isinstance(y, (tuple, list)):
      init_grad = tuple(tf.ones_like(t) for t in y)
    else:
      init_grad = tf.ones_like(y)
    df = autodiff(
        func,
        motion=motion,
        optimized=optimized,
        preserve_result=preserve_result,
        wrt=wrt,
        verbose=True)
    if motion == 'joint':
      # TODO: This won't work if func has default args unspecified.
      dx = df(*args + (init_grad,))
    else:
      dx = df(*args, init_grad=init_grad)
    return tensors_to_numpy(dx)

  def reference_func():
    gradval = tensors_to_numpy(tfe.gradients_function(func, params=wrt)(*args))
    if preserve_result:
      val = tensors_to_numpy(func(*args))
      if isinstance(gradval, (tuple)):
        return gradval + (val,)
      return gradval, val
    else:
      return gradval

  def backup_reference_func():
    func_ = as_numpy_sig(func)
    args_ = tensors_to_numpy(args)
    gradval = utils.numeric_grad(utils.numeric_grad(func_))(*args_)
    if preserve_result:
      val = tensors_to_numpy(func(*args))
      return gradval, val
    else:
      return gradval

  utils.assert_result_matches_reference(
      tangent_func, reference_func, backup_reference_func,
      # Some ops like tf.divide diverge significantly due to what looks like
      # numerical instability.
      tolerance=1e-5)
