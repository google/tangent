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
"""Going from AST or source code to executable code."""
from __future__ import absolute_import
import os
import tempfile
from uuid import uuid4

import gast
import six
if six.PY3:
  from importlib import util
else:
  import imp

from tangent import quoting


def compile_file(source, globals_=None):
  """Compile by saving to file and importing that.

  Compiling the AST/source code this way ensures that the source code is
  readable by e.g. `pdb` or `inspect`.

  Args:
    source: The code to compile, either as a string or as an AST.
    globals_: A dictionary of variables that should be available as globals in
        the compiled module. They will be monkey patched after importing the
        module.

  Returns:
    A module object containing the compiled source code.
  """
  if isinstance(source, gast.AST):
    source = quoting.to_source(source)

  # Write source to temporary file
  tempdir = tempfile.mkdtemp()
  uuid = str(uuid4().hex[:4])
  tmpname = os.path.join(tempdir, 'tangent_%s.py' % uuid)
  with open(tmpname, 'w') as f:
    f.write(source)

  # Load the temporary file as a module
  module_name = 'tangent_%s' % uuid
  if six.PY3:
    spec = util.spec_from_file_location(module_name, tmpname)
    m = util.module_from_spec(spec)
    spec.loader.exec_module(m)
  else:
    m = imp.load_source(module_name, tmpname)

  # Update the modules namespace
  if globals_:
    m.__dict__.update(globals_)
  return m


def compile_function(node, globals_=None):
  """Convert an AST or string into a function with inspectable source.

  This function uses `compile_file` internally, but instead of returning the
  entire module it will return the function only.

  Args:
    node: A `FunctionDef` node or a `Module` node which contains at least one
        `FunctionDef` node. If a module contains multiple functions, a handle
        to the first one will be returned.
    globals_: See `compile_file`

  Returns:
    A handle to the compiled function.

  Raises:
    TypeError: If the input is not a string or AST.
    ValueError: If no function can be found.
  """
  if not isinstance(node, gast.AST):
    if not isinstance(node, six.string_types):
      raise TypeError
    node = gast.parse(node)
  if isinstance(node, gast.Module):
    for succ in node.body:
      if isinstance(succ, gast.FunctionDef):
        name = succ.name
        break
    else:
      raise ValueError('no function found')
  elif isinstance(node, gast.FunctionDef):
    name = node.name
  else:
    raise TypeError
  module = compile_file(node, globals_)
  return getattr(module, name)
