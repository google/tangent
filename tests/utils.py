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
"""Common testing utilities."""
from copy import deepcopy

from autograd import grad as ag_grad
from autograd import value_and_grad as ag_value_and_grad
from autograd.misc.flatten import flatten
import autograd.numpy as ag_np
import numpy as np
import tangent

# Autograd's NumPy implementation may be missing the definition for _NoValue.
if not hasattr(ag_np, '_NoValue'):
  ag_np._NoValue = np._NoValue  # pylint: disable=protected-access


def assert_forward_not_implemented(func, wrt):
  try:
    tangent.autodiff(func, mode='forward', preserve_result=False, wrt=wrt)
    assert False, 'Remove this when implementing.'
  except NotImplementedError:
    pass


def _assert_allclose(a, b, tol=1e-5):
  if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
    for ia, ib in zip(a, b):
      _assert_allclose(ia, ib, tol)
  else:
    try:
      a = np.nan_to_num(a)
      b = np.nan_to_num(b)
      assert np.allclose(a, b, tol), ('Expected: %s\nGot: %s' % (b, a))
    except TypeError:
      raise TypeError('Could not compare values %s and %s' % (a, b))


def assert_result_matches_reference(
    tangent_func,
    reference_func,
    backup_reference_func,
    tolerance=1e-7):
  """Test Tangent functionality against reference implementation.

  Args:
    tangent_func: Returns the Tangent derivative.
    reference_func: Returns the derivative calculated by the reference
        implementation.
    backup_reference_func: Returns the derivative calculated by a catch-all
        implementation, should the reference be unavailable.
    tolerance: Absolute tolerance override for FP comparisons.
  """
  tangent_value = tangent_func()
  try:
    reference_value = reference_func()
  except (ImportError, TypeError) as e:
    if __debug__:
      print('WARNING: Reference function call failed. The test will revert to '
            'the backup reference.\nReason for failure: %s' % e)
    # TODO: Try to narrow the exception handler.
    reference_value = backup_reference_func()
  _assert_allclose(tangent_value, reference_value, tolerance)


def numeric_grad(func, eps=1e-6):
  """Generate a finite-differences gradient of function `f`.

  def f(x, *args):
    ...
    return scalar

  g = numeric_grad(f, eps=1e-4)
  finite_difference_grad_of_x = g(x, *args)

  Adapted from github.com/hips/autograd
  """
  def g(x, *args):
    fd_grad, unflatten_fd = flatten(tangent.init_grad(x))
    y = func(deepcopy(x), *args)
    seed = np.ones_like(y)
    for d in range(fd_grad.size):
      x_flat, unflatten_x = flatten(deepcopy(x))
      x_flat[d] += eps / 2
      a = np.array(func(unflatten_x(x_flat), *args))
      x_flat, unflatten_x = flatten(deepcopy(x))
      x_flat[d] -= eps / 2
      b = np.array(func(unflatten_x(x_flat), *args))
      fd_grad[d] = np.dot((a - b) / eps, seed)
    return unflatten_fd(fd_grad)

  return g


def test_reverse_array(func, motion, optimized, preserve_result, *args):
  """Test gradients of functions with NumPy-compatible signatures."""

  def tangent_func():
    y = func(*deepcopy(args))
    if np.array(y).size > 1:
      init_grad = np.ones_like(y)
    else:
      init_grad = 1
    func.__globals__['np'] = np
    df = tangent.autodiff(
        func,
        mode='reverse',
        motion=motion,
        optimized=optimized,
        preserve_result=preserve_result,
        verbose=1)
    if motion == 'joint':
      return df(*deepcopy(args) + (init_grad,))
    return df(*deepcopy(args), init_grad=init_grad)

  def reference_func():
    func.__globals__['np'] = ag_np
    if preserve_result:
      val, gradval = ag_value_and_grad(func)(*deepcopy(args))
      return gradval, val
    else:
      return ag_grad(func)(*deepcopy(args))

  def backup_reference_func():
    func.__globals__['np'] = np
    df_num = numeric_grad(func)
    gradval = df_num(*deepcopy(args))
    if preserve_result:
      val = func(*deepcopy(args))
      return gradval, val
    else:
      return gradval

  assert_result_matches_reference(tangent_func, reference_func,
                                  backup_reference_func)


def test_forward_array(func, wrt, preserve_result, *args):
  """Test derivatives of functions with NumPy-compatible signatures."""

  def tangent_func():
    func.__globals__['np'] = np
    df = tangent.autodiff(
        func,
        mode='forward',
        preserve_result=preserve_result,
        wrt=wrt,
        optimized=True,
        verbose=1)
    args_ = args + (1.0,)  # seed gradient
    return df(*deepcopy(args_))

  def reference_func():
    func.__globals__['np'] = ag_np
    if preserve_result:
      # Note: ag_value_and_grad returns (val, grad) but we need (grad, val)
      val, gradval = ag_value_and_grad(func)(*deepcopy(args))
      return gradval, val
    else:
      return ag_grad(func)(*deepcopy(args))

  def backup_reference_func():
    func.__globals__['np'] = np
    df_num = numeric_grad(func)
    gradval = df_num(*deepcopy(args))
    if preserve_result:
      val = func(*deepcopy(args))
      return gradval, val
    else:
      return gradval

  assert_result_matches_reference(tangent_func, reference_func,
                                  backup_reference_func)
