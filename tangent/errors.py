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
"""Tangent-specific errors."""
from __future__ import absolute_import


class TangentParseError(SyntaxError):
  """Error generated when encountering an unsupported feature."""
  pass


class ForwardNotImplementedError(NotImplementedError):
  """Error generated when encountering a @tangent_ yet to be implemented."""

  def __init__(self, func):
    NotImplementedError.__init__(
        self, 'Forward mode for function "%s" is not yet implemented.' %
        func.__name__)


class ReverseNotImplementedError(NotImplementedError):
  """Error generated when encountering an @adjoint yet to be implemented."""

  def __init__(self, func):
    NotImplementedError.__init__(
        self,
        'Reverse mode for function "%s" is not yet implemented.' %
        func.__name__,
    )
