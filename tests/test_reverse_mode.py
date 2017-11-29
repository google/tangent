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
"""Reverse-mode tests.

Notes
-----
Arguments func, a, b, c, x, and n are automatically filled in.

Pass --short for a quick run.

"""
import sys

from autograd import grad as ag_grad
from autograd.misc.flatten import flatten
import autograd.numpy as ag_np
import numpy as np
import pytest

import tangent
from tangent.grad_util import INPUT_DERIVATIVE
from tangent import quoting
import tfe_utils
import utils
from functions import bilinear
from functions import dict_saxpy
from functions import inlining_contextmanager
from functions import logistic_regression
from functions import nested_dict
from functions import rnn
from functions import unpacking_args_saxpy


def test_parses(func):
  """Test all functions parse."""
  quoting.parse_function(func)


def test_logistic_regression(motion, optimized):
  func = logistic_regression
  w = np.random.randn(3, 5)
  b = np.random.randn(5)
  input_ = np.random.rand(3)
  label = np.zeros(5)
  label[1] = 1

  func.__globals__['np'] = np
  df = tangent.autodiff(
      func,
      wrt=(2, 3),
      motion=motion,
      optimized=optimized,
      verbose=True,
      input_derivative=INPUT_DERIVATIVE.DefaultOne)
  dw, db = df(input_, label, w, b)

  func.__globals__['np'] = ag_np
  ag_dw = ag_grad(func, argnum=2)(input_, label, w, b)
  ag_db = ag_grad(func, argnum=3)(input_, label, w, b)
  assert np.allclose(ag_dw, dw)
  assert np.allclose(ag_db, db)


def test_rnn(motion, optimized):
  func = rnn
  w = np.random.randn(2, 3)
  inputs = np.random.randn(3, 2)

  func.__globals__['np'] = np
  df = tangent.autodiff(
      func,
      wrt=(0, 1),
      motion=motion,
      optimized=optimized,
      verbose=True,
      input_derivative=INPUT_DERIVATIVE.DefaultOne)
  dinputs, dw = df(inputs, w)

  num_dinputs = utils.numeric_grad(func)(inputs, w)
  num_dw = utils.numeric_grad(lambda w, x: func(x, w))(w, inputs)
  assert np.allclose(num_dw, dw)
  assert np.allclose(num_dinputs, dinputs)


def test_bilinear(optimized):
  func = bilinear
  D = 3
  np.random.seed(0)
  x = np.random.randn(1, D)
  h = np.random.randn(1, D)
  U = np.random.randn(D, D)
  w = np.random.randn(D, D)
  b = np.random.randn(1, D)

  func.__globals__['np'] = np
  df = tangent.autodiff(
      func,
      wrt=(0,),
      motion='joint',
      optimized=optimized,
      verbose=True,
      input_derivative=INPUT_DERIVATIVE.DefaultOne)
  dx = df(x, h, U, w, b)

  num_dx = utils.numeric_grad(func)(x, h, U, w, b)
  assert np.allclose(num_dx, dx)


def test_attributes():
  def f(x):
    return x.shape
  try:
    utils.test_reverse_array(f, 'JOINT', False, False, np.array([1.0, 2.0]))
    assert False
  except ValueError as expected:
    assert 'attributes are not yet supported' in str(expected)


def test_grad_unary(func, motion, optimized, preserve_result, a):
  """Test gradients of single-argument scalar functions."""
  utils.test_reverse_array(func, motion, optimized, preserve_result, a)


def test_grad_binary(func, motion, optimized, preserve_result, a, b):
  """Test gradients of two-argument scalar functions."""
  utils.test_reverse_array(func, motion, optimized, preserve_result, a, b)


def test_grad_ternary(func, motion, optimized, preserve_result, a, b, c):
  """Test gradients of three-argument scalar functions."""
  utils.test_reverse_array(func, motion, optimized, preserve_result, a, b, c)


def test_grad_vector(func, motion, optimized, preserve_result, x):
  """Test gradients of vector functions."""
  utils.test_reverse_array(func, motion, optimized, preserve_result, x)


def test_grad_square_matrix(func, motion, optimized, preserve_result, sqm):
  """Test gradients of square matrix functions."""
  utils.test_reverse_array(func, motion, optimized, preserve_result, sqm)


def test_grad_binary_int(func, motion, optimized, preserve_result, a, n):
  """Test gradients of functions with scalar and integer input."""
  utils.test_reverse_array(func, motion, optimized, preserve_result, a, n)


def test_inlining_contextmanager(motion, optimized, a):
  func = inlining_contextmanager
  func = tangent.tangent(func)

  func.__globals__['np'] = np
  df = tangent.autodiff(
      func,
      motion=motion,
      optimized=optimized,
      verbose=True,
      input_derivative=INPUT_DERIVATIVE.DefaultOne)
  dx = df(a)

  func.__globals__['np'] = ag_np
  df_ag = ag_grad(func)
  df_ag(a)
  assert np.allclose(dx, 2.9 * a**2)


def test_dict_saxpy(motion, optimized, a, b, c):
  func = dict_saxpy
  func = tangent.tangent(func)

  func.__globals__['np'] = np
  df = tangent.autodiff(
      func,
      motion=motion,
      optimized=optimized,
      verbose=True,
      input_derivative=INPUT_DERIVATIVE.DefaultOne)
  dx = df(dict(a=a, b=b, c=c))

  df_num = utils.numeric_grad(func)
  dx_num = df_num(dict(a=float(a), b=float(b), c=float(c)))
  flat_dx, _ = flatten(dx)
  flat_dx_num, _ = flatten(dx_num)
  assert np.allclose(flat_dx, flat_dx_num)


def test_unpacking_args_saxpy(motion, optimized, a, b, c):
  func = unpacking_args_saxpy
  func = tangent.tangent(func)

  func.__globals__['np'] = np
  df = tangent.autodiff(
      func,
      motion=motion,
      optimized=optimized,
      verbose=True,
      input_derivative=INPUT_DERIVATIVE.DefaultOne)
  dx = df((a, b, c))

  df_num = utils.numeric_grad(func)
  dx_num = df_num((a, b, c))
  assert np.allclose(dx, dx_num)


def test_nested_dict(motion, optimized):
  p = dict(i=dict(j=3.0, k=4.0))
  func = nested_dict
  df = tangent.autodiff(
      func,
      motion=motion,
      optimized=optimized,
      verbose=True,
      input_derivative=INPUT_DERIVATIVE.DefaultOne)
  dx = df(p)

  df_ag = ag_grad(func)
  dx_ag = df_ag(p)
  for k in p['i']:
    assert np.allclose(dx['i'][k], dx_ag['i'][k])


def test_grad_unary_tensor(func, motion, optimized, preserve_result, t):
  """Test gradients of functions with single tensor input."""
  tfe_utils.test_rev_tensor(func, motion, optimized, preserve_result, (0,), t)


def test_grad_unary_reduction(func, motion, optimized, preserve_result,
                              timage, boolean):
  """Test gradients of reduction functions."""
  tfe_utils.test_rev_tensor(func, motion, optimized, preserve_result, (0,),
                            timage, boolean)


def test_grad_binary_tensor(func, motion, optimized, preserve_result, t1, t2):
  """Test gradients of functions with binary tensor inputs."""
  tfe_utils.test_rev_tensor(func, motion, optimized, preserve_result, (0, 1),
                            t1, t2)


def test_grad_matmul(func, motion, optimized, preserve_result, mat1, mat2,
                     boolean1, boolean2):
  """Test gradients of functions with binary matrix inputs."""
  tfe_utils.test_rev_tensor(func, motion, optimized, preserve_result, (0, 1),
                            mat1, mat2, boolean1, boolean2)


def test_grad_matmul_higherdim(func, motion, optimized, preserve_result,
                               timage1, timage2, boolean1, boolean2):
  """Test gradients of functions with binary matrix inputs."""
  tfe_utils.test_rev_tensor(func, motion, optimized, preserve_result, (0, 1),
                            timage1, timage2, boolean1, boolean2)


def test_grad_tensor_broadcast(func, motion, optimized, preserve_result, s,
                               t):
  """Test gradients of functions with binary tensor inputs."""
  tfe_utils.test_rev_tensor(func, motion, optimized, preserve_result, (0, 1),
                            s, t)


def test_grad_image(func, motion, optimized, preserve_result, timage, tkernel,
                    conv2dstrides):
  """Test gradients of image functions."""
  # TODO: Upgrade utils.py to allow simultaneous testing of uneven args.
  tfe_utils.test_rev_tensor(func, motion, optimized, preserve_result, (0,),
                            timage, tkernel, conv2dstrides)
  tfe_utils.test_rev_tensor(func, motion, optimized, preserve_result, (1,),
                            timage, tkernel, conv2dstrides)


def test_grad_image_pooling(func, motion, optimized, preserve_result, timage,
                            pool2dsizes, conv2dstrides):
  tfe_utils.test_rev_tensor(func, motion, optimized, preserve_result, (0,),
                            timage, pool2dsizes, conv2dstrides)


if __name__ == '__main__':
  assert not pytest.main([__file__, '--short'] + sys.argv[1:])
