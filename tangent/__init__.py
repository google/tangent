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
"""Several imports to flatten the Tangent namespace for end users."""
from __future__ import absolute_import
import functools

import gast

from tangent import annotate
from tangent import ast as ast_
from tangent import compile as compile_
from tangent.tracing import trace
from tangent.tracing import trace_grad
from tangent.utils import add_grad
from tangent.utils import array_size
from tangent.utils import astype
from tangent.utils import balanced_eq
from tangent.utils import copy
from tangent.utils import grad_dot
from tangent.utils import init_grad
from tangent.utils import insert_grad_of
from tangent.utils import pop
from tangent.utils import pop_stack
from tangent.utils import push
from tangent.utils import push_stack
from tangent.utils import shapes_match
from tangent.utils import Stack
from tangent.utils import unbroadcast
from tangent.utils import unreduce
from tangent.utils import unreduce_like

# Imported last to avoid circular imports
from tangent.grad_util import grad, autodiff, vjp, jvp
from tangent.errors import *
from tangent.tf_extensions import *


class RemoveWith(gast.NodeTransformer):
  """A transformer that removes `with insert_grad_of` statements."""

  def visit_With(self, node):
    if ast_.is_insert_grad_of_statement(node):
      return None
    else:
      return node


def tangent(f):
  """A decorator which removes the `with insert_grad_of` statement.

  This allows the function to be called as usual.

  Args:
    f: A function

  Returns:
    A function with any `with insert_grad_of` context managers removed.
  """
  node = annotate.resolve_calls(f)
  RemoveWith().visit(node)
  wrapped = functools.wraps(f)(compile_.compile_function(node))
  wrapped.tangent = f
  return wrapped
