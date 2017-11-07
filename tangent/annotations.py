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
"""Handling annotations on AST nodes."""
from __future__ import absolute_import

import gast

ANNOTATION_FIELD = '_tangent'
# These annotation's won't be cleared between passes
FIXED_ANNOTATIONS = set(['pop', 'push', 'add_grad', 'init_grad', 'pri', 'adj',
                         'push_func', 'pop_func', 'adjoint_var',
                         'temp_adjoint_var', 'temp_var', 'pri_call',
                         'adj_call', 'comment', 'pre_anf'])


def setanno(node, key, value, safe=True):
  annotations = getattr(node, ANNOTATION_FIELD, {})
  setattr(node, ANNOTATION_FIELD, annotations)
  if safe and hasanno(node, key):
    raise ValueError('annotation already present')
  annotations[key] = value

  # So that the annotations survive gast_to_ast() and ast_to_gast()
  if ANNOTATION_FIELD not in node._fields:
    node._fields += (ANNOTATION_FIELD,)


def hasanno(node, key):
  annotations = getattr(node, ANNOTATION_FIELD, {})
  return key in annotations


def setdefaultanno(node, key, value=None):
  if not hasanno(node, key):
    setanno(node, key, value)
  return getanno(node, key)


def clearanno(node):
  for succ in gast.walk(node):
    if hasattr(succ, ANNOTATION_FIELD):
      new = {}
      for anno in FIXED_ANNOTATIONS:
        if hasanno(succ, anno):
          new[anno] = getanno(succ, anno)
      setattr(succ, ANNOTATION_FIELD, new)
  return node


def getanno(node, key, default=None):
  annotations = getattr(node, ANNOTATION_FIELD, {})
  if key not in annotations and default is None:
    raise KeyError('Node "%s" has no annotation "%s"' % (node, key))
  return annotations.get(key, default)


def delanno(node, key):
  annotations = getattr(node, ANNOTATION_FIELD, {})
  del annotations[key]
