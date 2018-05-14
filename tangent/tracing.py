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
"""Utilities for tracing code, a useful fallback when ahead-of-time AD fails.
"""


class Traceable(object):
  pass


def trace_grad(fn, args):
  """Trace a function, and return a VJP and the function's output."""
  from tensorflow.python.eager.backprop import make_vjp
  result, vjp = make_vjp(fn)(*args)
  return result, vjp


def trace(fn):
  """Decorator that marks a function to be traced."""
  fn.should_trace = True
  return fn
