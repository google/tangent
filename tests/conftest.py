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
"""Automatically test gradients with multiple inputs, modes and motions."""
import numpy as np
import six

import functions
import tfe_utils


blacklisted = [
    'inlining_contextmanager',
    'listcomp',
    'cart2polar',
    'iterpower_with_nested_def',
    'fn_multiple_return',
    'insert_grad_of',
    '_trace_mul',
    '_nontrace_mul',
    'active_subscript',  # TODO: fix then remove from blacklist
    'init_array_grad_maybe_active',  # TODO: fix then remove from blacklist
]

funcs = [f for f in functions.__dict__.values() if callable(f)]
whitelist = [f for f in funcs if f.__name__ not in blacklisted]
blacklist = [f for f in funcs if f.__name__ in blacklisted]


def pytest_addoption(parser):
  # Only test with one input
  parser.addoption('--short', action='store_true')
  # Only test with all inputs
  parser.addoption('--all', action='store_true')
  # Restrict to certain functions by name
  parser.addoption('--func_filter', action='store')


def pytest_generate_tests(metafunc):
  # Parametrize the functions
  if 'func' in metafunc.fixturenames:
    func_filter = metafunc.config.option.func_filter

    # Test takes args, only pass funcs with same signature
    args = tuple(
        arg for arg in metafunc.fixturenames
        if arg not in ('func', 'motion', 'optimized', 'preserve_result'))
    if args:
      func_args = []
      for f in whitelist:
        fc = six.get_function_code(f)
        if fc.co_varnames[:fc.co_argcount] == args:
          func_args.append(f)
    else:
      func_args = funcs

    if func_filter:
      func_args = [f for f in func_args if func_filter in f.__name__]

    func_names = [f.__name__ for f in func_args]
    metafunc.parametrize('func', func_args, ids=func_names)

  if 'motion' in metafunc.fixturenames:
    metafunc.parametrize('motion', ('split', 'joint'))

  if 'optimized' in metafunc.fixturenames:
    metafunc.parametrize('optimized', (True, False),
                         ids=('optimized', 'unoptimized'))

  if 'preserve_result' in metafunc.fixturenames:
    metafunc.parametrize('preserve_result', (True, False))

  # Parametrize the arguments
  short = metafunc.config.option.short

  bools = [True, False]
  for arg in ['boolean', 'boolean1', 'boolean2']:
    if arg in metafunc.fixturenames:
      metafunc.parametrize(arg, bools)

  scalars = [2.] if short else [
      -2., -1.5, -1., -0.5, -0.1, 0.1, 0.5, 1., 1.5, 2.
  ]
  for arg in 'abc':
    if arg in metafunc.fixturenames:
      metafunc.parametrize(arg, scalars)

  integers = [1] if short else [1, 2, 3]
  if 'n' in metafunc.fixturenames:
    metafunc.parametrize('n', integers)

  vectors = [np.random.randn(i) for i in ((3,) if short else (3, 5, 10))]
  if 'x' in metafunc.fixturenames:
    metafunc.parametrize('x', vectors)

  square_matrices = [np.random.randn(*i) for i in (((3, 3),) if short else ((1, 1), (5, 5)))]
  if 'sqm' in metafunc.fixturenames:
    metafunc.parametrize('sqm', square_matrices)

  tfe_utils.register_parametrizations(metafunc, short)
