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
"""Moving between source code and AST."""
from __future__ import absolute_import
import inspect
import textwrap

import astor
import gast

from tangent import annotations as anno


class TangentParseError(SyntaxError):
  pass


class SourceWithCommentGenerator(astor.codegen.SourceGenerator):
  """Source code generator that outputs comments."""

  def __init__(self, *args, **kwargs):
    super(SourceWithCommentGenerator, self).__init__(*args, **kwargs)
    self.new_indentation = True

  def body(self, statements):
    self.new_indentation = True
    super(SourceWithCommentGenerator, self).body(statements)

  def visit(self, node, abort=astor.codegen.SourceGenerator.abort_visit):
    if anno.hasanno(node, 'comment'):
      comment = anno.getanno(node, 'comment')
      # Preprocess the comment to fit to maximum line width of 80 characters
      linewidth = 78
      if comment['location'] in ('above', 'below'):
        comment['text'] = comment['text'][:linewidth]
      n_newlines = 1 if self.new_indentation else 2
      if comment['location'] == 'above':
        self.result.append('\n' * n_newlines)
        self.result.append(self.indent_with * self.indentation)
        self.result.append('# %s' % comment['text'])
        super(SourceWithCommentGenerator, self).visit(node)
      elif comment['location'] == 'below':
        super(SourceWithCommentGenerator, self).visit(node)
        self.result.append('\n')
        self.result.append(self.indent_with * self.indentation)
        self.result.append('# %s' % comment['text'])
        self.result.append('\n' * (n_newlines - 1))
      elif comment['location'] == 'right':
        super(SourceWithCommentGenerator, self).visit(node)
        self.result.append(' # %s' % comment['text'])
      else:
        raise TangentParseError('Only valid comment locations are '
                                'above, below, right')
    else:
      self.new_indentation = False
      super(SourceWithCommentGenerator, self).visit(node)


def to_source(node, indentation=' ' * 4):
  """Return source code of a given AST."""
  if isinstance(node, gast.AST):
    node = gast.gast_to_ast(node)
  generator = SourceWithCommentGenerator(indentation, False,
                                         astor.string_repr.pretty_string)
  generator.visit(node)
  generator.result.append('\n')
  return astor.source_repr.pretty_source(generator.result).lstrip()


def parse_function(fn):
  """Get the source of a function and return its AST."""
  try:
    return parse_string(inspect.getsource(fn))
  except (IOError, OSError) as e:
    raise ValueError(
        'Cannot differentiate function: %s. Tangent must be able to access the '
        'source code of the function. Functions defined in a Python '
        'interpreter and functions backed by C extension modules do not '
        'have accessible source code.' % e)


def parse_string(src):
  """Parse a string into an AST."""
  return gast.parse(textwrap.dedent(src))


def quote(src_string, return_expr=False):
  """Go from source code to AST nodes.

  This function returns a tree without enclosing `Module` or `Expr` nodes.

  Args:
    src_string: The source code to parse.
    return_expr: Whether or not to return a containing expression. This can be
        set to `True` if the result is to be part of a series of statements.

  Returns:
    An AST of the given source code.

  """
  node = parse_string(src_string)
  body = node.body
  if len(body) == 1:
    if isinstance(body[0], gast.Expr) and not return_expr:
      out = body[0].value
    else:
      out = body[0]
  else:
    out = node
  return out


def unquote(node):
  """Go from an AST to source code."""
  return to_source(node).strip()
