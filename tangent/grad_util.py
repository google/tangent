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
"""Utilities to take derivatives of Python functions.

For notation and theory, please refer to Chapter 3 of "Evaluating Derivatives"
by Griewank and Walther.

We expose a few APIs to do this.
- grad(f): generate gradient of a function f : R^n -> R. Scalar output will be
checked.
- autodiff(f, mode='forward'): generate the forward-mode derivative of a
function f.
    Best for functions f: R^n -> R^m where m >> n
- autodiff(f, mode='reverse'): generate the reverse-mode derivative of a
function f.
  Best for functions f: R^n -> R^m where n >> m.


Forward-mode and reverse-mode are the two main ways to calculate derivatives.
Given a function `F`, with two arguments:

```
Z = F(X, Y)
```

we can calculate in "forward mode", which returns

```
# Forward mode
# dZ, given dX, X, Y
dZ = dF(X, Y, dX))
```

or, we can calculate the derivative of the in "reverse mode", which returns
```
# Reverse mode
# bX, given bZ, X, Y
bX = bF(X, Y, bZ)
```

"""
from __future__ import absolute_import

import enum
import inspect
import gast
import numpy
import six
from tangent import anf as anf_
from tangent import annotate
from tangent import ast as ast_
from tangent import comments
from tangent import compile as compile_
from tangent import fence
from tangent import forward_ad
from tangent import naming
from tangent import optimization
from tangent import quoting
from tangent import reverse_ad

INPUT_DERIVATIVE = enum.Enum('InputDerivative',
                             ('Required', 'DefaultOne', 'DefaultOnes'))


def autodiff_ast(func, wrt, motion, mode, preserve_result, check_dims, verbose):
  """Perform AD on a single function and return the AST.

  Args:
    See `grad`.

  Returns:
    node: The AST of a module containing the adjoint and primal function
        definitions.
    required: A list of non-built in functions that this function called, and
        of which the primals and adjoints need to be made available in order
        for the returned function to run.
  """
  node = annotate.resolve_calls(func)
  fence.validate(node, inspect.getsource(func))
  node = anf_.anf(node)
  if verbose >= 2:
    print('ANF')
    print(quoting.to_source(node))
  if mode == 'reverse':
    node, required, stack = reverse_ad.reverse_ad(node.body[0], wrt,
                                                  preserve_result, check_dims)
    if verbose >= 2:
      print('RAW')
      print(quoting.to_source(node))
    if motion == 'split':
      node = reverse_ad.split(node, stack)
    else:
      node = reverse_ad.joint(node)
    if verbose >= 2:
      print('MOTION')
      print(quoting.to_source(node))
  elif mode == 'forward':
    node, required = forward_ad.forward_ad(node.body[0], wrt, preserve_result,
                                           check_dims)
  return node, required


def autodiff_tree(func, wrt, motion, mode, preserve_result, check_dims,
                  verbose):
  """Perform AD on all functions in a call tree.

  This function walks the call tree and differentiates each function in it. It
  also ensures that the global namespaces that each function in the call tree
  was in are merged.

  The `tangent` and `numpy` packages are added to the namespace here, so that
  the gradient templates can assume that they are present.

  Args:
    See `grad`.

  Returns:
    final: A single module which contains the primals and adjoints of all the
        functions in the call tree.
    namespace: A merged dictionary with all the variables in the global
        namespaces of each function. The primals and adjoints need access to
        these in order to execute.
  """
  # Imported here to avoid circular imports
  import tangent
  namespace = {'tangent': tangent, 'numpy': numpy}

  done = set()
  final = gast.Module(body=[])
  namespace.update(six.get_function_globals(func))

  node, required = autodiff_ast(func, wrt, motion, mode, preserve_result,
                                check_dims, verbose)
  final.body.extend(node.body)

  to_do = set(required)
  if motion == 'split' and mode == 'reverse':
    done.add((func, wrt))
    to_do -= done

  while to_do:
    func, wrt = to_do.pop()
    namespace.update(six.get_function_globals(func))

    node, required = autodiff_ast(
        func=func,
        wrt=wrt,
        motion='split',
        mode=mode,
        preserve_result=True,
        check_dims=False,
        verbose=verbose)

    final.body.extend(node.body)
    done.add((func, wrt))
    to_do.update(required)
    to_do -= done

  return final, namespace


def vjp(func,
        wrt=(0,),
        optimized=True,
        check_dims=True,
        preserve_result=False,
        verbose=0):
  """Convenience function to produce vector-Jacobian products.

  See `autodiff` for function arguments.
  Uses reverse-mode joint-motion autodiff to produce the VJP.
  """
  return autodiff(
      func,
      wrt=wrt,
      motion='joint',
      mode='reverse',
      optimized=optimized,
      preserve_result=preserve_result,
      input_derivative=INPUT_DERIVATIVE.Required,
      check_dims=check_dims,
      verbose=verbose)


def jvp(func,
        wrt=(0,),
        optimized=True,
        check_dims=True,
        preserve_result=False,
        verbose=0):
  """Convenience function to produce Jacobian-vector products.

  See `autodiff` for function arguments.
  Uses forward-mode autodiff to produce the JVP.
  """
  return autodiff(
      func,
      wrt=wrt,
      mode='forward',
      optimized=optimized,
      preserve_result=preserve_result,
      input_derivative=INPUT_DERIVATIVE.Required,
      check_dims=check_dims,
      verbose=verbose)


def autodiff(func,
             wrt=(0,),
             optimized=True,
             motion='joint',
             mode='reverse',
             preserve_result=False,
             check_dims=True,
             input_derivative=INPUT_DERIVATIVE.Required,
             verbose=0):
  """Build the vector-Jacobian or Jacobian-vector product of a function `func`.

  For a vector-Jacobian product (reverse-mode autodiff):
  This function proceeds by finding the primals and adjoints of all the
  functions in the call tree.
  For a Jacobian-vector product (forward-mode autodiff):
  We first find the primals and tangents of all functions in the call tree.

  It then wraps the top level function (i.e. the
  one passed as `func`) in a slightly more user-friendly interface. It then
  compiles the function and attaches to it the global namespace it needs to
  run.

  Args:
    func: The function to take the gradient of.
    wrt: A tuple of argument indices to differentiate with respect to. By
        default the derivative is taken with respect to the first argument.
    optimized: Whether to optimize the gradient function (`True` by default).
    motion: Either 'split' (separate functions for forward and backward pass)
        or 'joint' motion (a single combined function). Joint mode is the
        default.
    mode: Either 'forward' or 'reverse' mode. Forward mode is more efficient
        when the input dimensionality is lower than the output dimensionality,
        whereas it is the opposite for reverse mode.
    input_derivative: An enum indicating whether the user must supply an input
        derivative, and if not, what the default value is. See the
        possible values of INPUT_DERIVATIVE in this file.

    preserve_result: A boolean indicating whether or not the generated gradient
        function should also return the output of the original function.
        If False, the return signature of the input and output functions will be
        > val = func(*args)
        > df = grad(func,preserve_result=False)
        > gradval = df(*args)
        If True,
        > val = func(*args)
        > df = grad(func,preserve_result=True)
        > gradval, val = func(*args)
        Note that if taking gradients with respect to multiple arguments,
        the primal value will be appended to the return signature. Ex:
        > val = func(x,y)
        > df = grad(func,wrt=(0,1),preserve_result=True)
        > dx,dy,val = df(x,y)

    verbose: If 1 the source code of the generated functions will be
        output to stdout at various stages of the process for debugging
        purposes. If > 1, all intermediate code generation steps will print.

  Returns:
    df: A function that calculates a derivative (see file-level documentation
    above
        for the kinds of derivatives available) with respect to arguments
        specified in `wrt`, using forward or reverse mode according to `mode`.
        If using reverse mode, the gradient is calculated in either split
        or joint motion according to the value passed in `motion`. If
        `preserve_result` is True, the function will also return the original
        result of `func`.
  """
  # If the function had the with insert_grad_of statements removed, retrieve them
  func = getattr(func, 'tangent', func)

  # Generate the derivative
  node, namespace = autodiff_tree(func, wrt, motion, mode, preserve_result,
                                  check_dims, verbose)

  if mode == 'reverse' and motion == 'joint':
    # Pull the stack definition and initial gradient into the function body
    # TODO: Use first FunctionDef instead of first element
    node.body[0] = _create_joint(node.body[0], func, wrt, input_derivative)
    if verbose >= 2:
      print('INLINED')
      print(quoting.to_source(node))
  if mode == 'forward':
    node = _create_forward(node)
  if optimized:
    # Optimize the resulting functions
    node = optimization.optimize(node)
  node = comments.remove_repeated_comments(node)
  if verbose >= 1:
    print(quoting.to_source(node))

  # Compile and return
  module = compile_.compile_file(node, namespace)
  if mode == 'forward' or motion == 'joint':
    return getattr(module, node.body[0].name)
  else:
    # Compiling the top-level function in split mode makes no sense, but we use
    # it for testing; hence we don't care about the source being readable
    forward = getattr(module, node.body[0].name)
    backward = getattr(module, node.body[1].name)

    # Imported here to avoid circular imports
    import tangent

    def df(*args, **kwargs):
      _stack = tangent.Stack()
      init_grad = kwargs.pop('init_grad', 1.0)
      forward(_stack, *args, **kwargs)
      dx = backward(_stack, init_grad, *args, **kwargs)
      if len(dx) == 1:
        dx, = dx
      return dx

    return df


def grad(func,
         wrt=(0,),
         optimized=True,
         preserve_result=False,
         check_dims=True,
         verbose=0):
  """Return the gradient of a function `func`.
  Args:
    func: The function to take the gradient of.
    wrt: A tuple of argument indices to differentiate with respect to. By
        default the derivative is taken with respect to the first argument.
    optimized: Whether to optimize the gradient function (`True` by default).
    preserve_result: A boolean indicating whether or not the generated gradient
        function should also return the output of the original function.
        If False, the return signature of the input and output functions will be
        > val = func(*args)
        > df = grad(func,preserve_result=False)
        > gradval = df(*args)
        If True,
        > val = func(*args)
        > df = grad(func,preserve_result=True)
        > gradval, val = func(*args)
        Note that if taking gradients with respect to multiple arguments,
        the primal value will be appended to the return signature. Ex:
        > val = func(x,y)
        > df = grad(func,wrt=(0,1),preserve_result=True)
        > dx,dy,val = df(x,y)
    check_dims: A boolean (`True` by default) indicating whether to check
        that the result of the original function `func` is a scalar, raising
        an error if it is not.
        Gradients are only valid for scalar-valued outputs, so we check
        this by defualt.
    verbose: If 1 the source code of the generated functions will be
        output to stdout at various stages of the process for debugging
        purposes. If > 1, all intermediate code generation steps will print.

  Returns:
    df: A function that calculates the gradient with respect to arguments
        specified in `wrt`, using forward or reverse mode according to `mode`.
        If using reverse mode, the gradient is calculated in either split
        or joint motion according to the value passed in `motion`. If
        `preserve_result` is True, the function will also return the original
        result of `func`.
  """
  return autodiff(
      func,
      wrt=wrt,
      motion='joint',
      mode='reverse',
      optimized=optimized,
      preserve_result=preserve_result,
      check_dims=check_dims,
      input_derivative=INPUT_DERIVATIVE.DefaultOne,
      verbose=verbose)


# TODO: these are utility functions, designed only for internal use.
# Should be moved to a separate file.
def _create_joint(fwdbwd, func, wrt, input_derivative):
  """Create a user-friendly gradient function.

  By default, gradient functions expect the stack to be passed to them
  explicitly. This function modifies the function so that the stack doesn't
  need to be passed and gets initialized in the function body instead.

  For consistency, gradient functions always return a tuple, even if the
  gradient of only one input was required. We unpack the tuple if it is of
  length one.

  Args:
    fwdbwd: An AST. The function definition of the joint primal and adjoint.
    func: A function handle. The original function that was differentiated.
    wrt: A tuple of integers. The arguments with respect to which we differentiated.

  Returns:
    The function definition of the new function.
  """
  # Correct return to be a non-tuple if there's only one element
  retval = fwdbwd.body[-1]
  if len(retval.value.elts) == 1:
    retval.value = retval.value.elts[0]

  # Make a stack init statement
  init_stack = quoting.quote('%s = tangent.Stack()' % fwdbwd.args.args[0].id)
  init_stack = comments.add_comment(init_stack, 'Initialize the tape')

  # Prepend the stack init to the top of the function
  fwdbwd.body = [init_stack] + fwdbwd.body

  # Replace the function arguments with the original ones
  grad_name = fwdbwd.args.args[1].id
  fwdbwd.args = quoting.parse_function(func).body[0].args

  # Give the function a nice name
  fwdbwd.name = naming.joint_name(func, wrt)

  # Allow the initial gradient to be passed as a keyword argument
  fwdbwd = ast_.append_args(fwdbwd, [grad_name])
  if input_derivative == INPUT_DERIVATIVE.DefaultOne:
    fwdbwd.args.defaults.append(quoting.quote('1.0'))
  return fwdbwd


def _create_forward(out_node):
  """Create a user-friendly forward function.

  Ensures that a single value instead of a tuple is returned if the user asked
  for the gradient with respect to only one input.

  Args:
    out_node: The function definition AST.

  Returns:
    The function definition with potentially changed return statement.
  """
  retval = out_node.body[0].body[-1]
  if len(retval.value.elts) == 1:
    retval.value = retval.value.elts[0]
  return out_node
