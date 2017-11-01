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
"""Handling comments on nodes.

To make the generated derivative source code more legible, statements are
annotated with human-readable comments.

"""
from __future__ import absolute_import

import gast

from tangent import annotations as anno


def add_comment(node, text, location='above'):
  """Add a comment to the given node.

  If the `SourceWithCommentGenerator` class is used these comments will be
  output as part of the source code.

  Note that a node can only contain one comment. Subsequent calls to
  `add_comment` will ovverride the existing comments.

  Args:
    node: The AST node whose containing statement will be commented.
    text: A comment string.
    location: Where the comment should appear. Valid values are 'above',
    'below' and 'right'

  Returns:
    The node with the comment stored as an annotation.
  """
  anno.setanno(node, 'comment', dict(location=location, text=text), safe=False)
  return node


def remove_repeated_comments(node):
  """Remove comments that repeat themselves.

  Multiple statements might be annotated with the same comment. This way if one
  of the statements is deleted during optimization passes, the comment won't be
  lost. This pass removes sequences of identical comments, leaving only the
  first one.

  Args:
    node: An AST

  Returns:
    An AST where comments are not repeated in sequence.

  """
  last_comment = {'text': None}
  for _node in gast.walk(node):
    if anno.hasanno(_node, 'comment'):
      comment = anno.getanno(_node, 'comment')
      if comment['text'] == last_comment['text']:
        anno.delanno(_node, 'comment')
      last_comment = comment
  return node
