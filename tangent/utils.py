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
"""Helper functions.

These are functions used at runtime by Tangent, and can be used to extend
Tangent to new data types.

TODO: Link to guide on how to extend the framework.
"""
from __future__ import absolute_import
from __future__ import division

from copy import copy as native_copy
import types

import autograd
import numpy
import six
from tangent import annotations as anno
from tangent import non_differentiable
from tangent import quoting


INIT_GRAD = quoting.quote('tangent.init_grad')
ADD_GRAD = quoting.quote('tangent.add_grad')
anno.setanno(INIT_GRAD, 'init_grad', True)
anno.setanno(ADD_GRAD, 'add_grad', True)


def array_size(x, axis):
  """Calculate the size of `x` along `axis` dimensions only."""
  axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
  return max(numpy.prod(axis_shape), 1)


class Stack(object):
  """A stack type that proxies list's `append` and `pop` methods.

  We don't use list directly so that we can test its type for the multiple-
  dispatch that occurs in `add_grad` and `init_grad`.
  """

  def __init__(self, vals=()):
    self._stack = list(vals)

  def append(self, x):
    self._stack.append(x)

  def pop(self):
    return self._stack.pop()

  def __len__(self):
    return len(self._stack)

  def __str__(self):
    return str(self._stack)

  def __repr__(self):
    return self._stack.__repr__()

# The values are binary functions with signature fn(array, like)
unbroadcasters = {
    numpy.ndarray:
        lambda array, like: unbroadcast_numpy_to(array, numpy.shape(like)),
    numpy.float32:
        lambda array, like: array,
    numpy.float64:
        lambda array, like: array,
    float:
        lambda array, like: array,
    int:
        lambda array, like: array,
    bool:
        lambda array, like: array,
}


def register_unbroadcast(t, unbroadcaster_function):
  """Register a new unbroadcaster.

  Unbroadcasters are used to undo broadcasting, e.g. np.eye(3) + 3
  will broadcast 3 to np.shape(np.eye(3)). In the backward pass, we have to
  undo this.

  Args:
    t: A Python type object. The data type supported by the
      unbroadcaster.
    unbroadcaster_function: A binary function that takes a first argument of
      type t, and a second argument that t needs to be unbroadcast to.
  """
  assert t not in unbroadcasters
  unbroadcasters[t] = unbroadcaster_function


def unbroadcast(array, like):
  """Reverse the broadcasting operation.

  Args:
    array: An array.
    like: An array that could have been broadcasted to the shape of array.

  Returns:
    Tensor with certain dimensions summed to match the shape of `like`.
  """
  unbroadcaster = unbroadcasters[type(array)]
  return unbroadcaster(array, like)


def create_unbroadcast_axis(shape, broadcast_shape):
  """Creates the reduction axis for unbroadcasting.

  Args:
    shape: A list. The shape after the broadcast operation.
    broadcast_shape: A list. The original shape the array being unbroadcast
      had.
  Returns:
    A list. The axes along which the array needs to be reduced. These axes will
    be distributed evenly into the original shape.
  """
  return tuple(
      -(1 + i)
      for i in range(len(broadcast_shape))
      if i >= len(shape) or broadcast_shape[-(1 + i)] > shape[-(1 + i)])


def unbroadcast_numpy_to(array, shape):
  """Reverse the broadcasting operation.

  Args:
    array: An array.
    shape: A shape that could have been broadcasted to the shape of array.

  Returns:
    Array with dimensions summed to match `shape`.
  """
  axis = create_unbroadcast_axis(shape, numpy.shape(array))
  return numpy.reshape(numpy.sum(array, axis=axis), shape)


def unreduce(array, shape, axis, keepdims):
  """Reverse summing over a dimension.

  Args:
    array: The array that was reduced.
    shape: The original shape of the array before reduction.
    axis: The axis or axes that were summed.
    keepdims: Whether these axes were kept as singleton axes.

  Returns:
    An array with axes broadcast to match the shape of the original array.
  """
  unreducer = unreducers[type(array)]
  return unreducer(array, shape, axis, keepdims)


def unreduce_like(array, original_array, axis, keepdims):
  """Reverse summing over a dimension.

  Args:
    array: The array that was reduced.
    original_array: An array whose shape to unreduce to.
    axis: The axis or axes that were summed.
    keepdims: Whether these axes were kept as singleton axes.

  Returns:
    An array with axes broadcast to match the shape of the original array.
  """
  atype = type(array)
  unreducer = unreducers[atype]
  shape = shape_functions[atype]
  return unreducer(array, shape(original_array), axis, keepdims)


def unreduce_array(array, shape, axis, keepdims):
  """Reverse summing over a dimension, NumPy implementation.

  Args:
    array: The array that was reduced.
    shape: The original shape of the array before reduction.
    axis: The axis or axes that were summed.
    keepdims: Whether these axes were kept as singleton axes.

  Returns:
    An array with axes broadcast to match the shape of the original array.
  """
  # NumPy uses a special default value for keepdims, which is equivalent to
  # False.
  if axis is not None and (not keepdims or keepdims is numpy._NoValue):  # pylint: disable=protected-access
    if isinstance(axis, int):
      axis = axis,
    for ax in sorted(axis):
      array = numpy.expand_dims(array, ax)
  return numpy.broadcast_to(array, shape)


# The values are unary functions.
shape_functions = {
    numpy.ndarray: numpy.shape,
    numpy.float32: numpy.shape,
    numpy.float64: numpy.shape,
    float: lambda _: (),
    int: lambda _: (),
    bool: lambda _: (),
}


def register_shape_function(t, shape_function):
  """Register a new shape function.

  Shape functions extract the shape of an array-like object.

  Args:
    t: A Python type object. The data type supported by the
      unreducer.
    shape_function: A unary function that returns a list or tuple with zero
      or more integers representing the dimensions of `t`.
  """
  assert t not in shape_functions
  shape_functions[t] = shape_function


# The values are functions with signature like `unreduce_array`
unreducers = {
    numpy.ndarray: unreduce_array,
    numpy.float32: unreduce_array,
    numpy.float64: unreduce_array,
    float: unreduce_array,
    int: unreduce_array,
    bool: unreduce_array,
}


def register_unreduce(t, unreducer_function):
  """Register a new unreducer.

  Unreducers are used to undo reduction, e.g. np.sum(np.eye(3))
  will reduce a (3,3) array to a scalar. In the backward pass, we have to
  undo this.

  Args:
    t: A Python type object. The data type supported by the
      unreducer.
    unreducer_function: A function with the same signature
      as e.g. `unreduce_array`
  """
  assert t not in unreducers
  unreducers[t] = unreducer_function


def astype(array, y):
  """A functional form of the `astype` method.

  Args:
    array: The array or number to cast.
    y: An array or number, as the input, whose type should be that of array.

  Returns:
    An array or number with the same dtype as `y`.
  """
  if isinstance(y, autograd.core.Node):
    return array.astype(numpy.array(y.value).dtype)
  return array.astype(numpy.array(y).dtype)


def balanced_eq(x, z, y):
  """Gradient of the max operator with tie breaking.

  Args:
    x: The left value
    z: The maximum of x and y
    y: The right value

  Returns:
    The gradient of the left value i.e. 1 if it is the maximum, 0.5 if they are
    equal and 0 if it was not the maximum.
  """
  return (x == z) / (1.0 + (x == y))


def init_common_object(obj):
  """Initialize gradients for the types of common objects we support."""
  if obj is numpy._globals._NoValue:  # pylint: disable=protected-access
    return obj
  raise ValueError('Unknown value to initialize: "%s"' % obj)


# The values are tuples (initializer, allow_lazy_initializer). If
# supports_lazy_initializer is true, Tangent may substitude actual instances
# of the object for the ZeroGradient wrapper, which is a lazy creator.
grad_initializers = {
    # TODO: We may be able to use ZeroGradient for ndarrays, too.
    numpy.ndarray: (numpy.zeros_like, False),
    numpy.float32: (lambda obj: 0.0, False),
    numpy.float64: (lambda obj: 0.0, False),
    list: (lambda obj: [init_grad(el) for el in obj], False),
    tuple: (lambda obj: [init_grad(el) for el in obj], False),
    dict: (lambda obj: {k: init_grad(v) for k, v in six.iteritems(obj)}, False),
    Stack: (lambda obj: Stack(), False),
    float: (lambda obj: 0.0, False),
    int: (lambda obj: 0, False),
    bool: (lambda obj: False, False),
}

if hasattr(types, 'ClassType'):
  grad_initializers[types.ClassType] = (init_common_object, False)
else:
  grad_initializers[type] = (init_common_object, False)


class ZeroGradient(object):
  """Lightweight substitute for zero gradients.

  This object may be used instead of an actual type when manipulating
  objects of the respective type is expensive.
  """

  def __init__(self, like):
    self._like = like

  def like(self):
    return self._like

  def instantiate(self):
    return grad_initializers[type(self._like)](self._like)


def register_init_grad(t, init_grad_function):
  """Register a new gradient initializer.

  Gradient initializers are used to initialize new adjoint and tangent
  variables.
  TODO: Link to the document explaining the overall terminology and mechanics.

  Args:
    t: A Python type object. The data type supported by the initializer.
    init_grad_function: A unary function that takes an argument of type t
      and returns a zero object of the same size as the argument. For example,
      the gradient initializer for Numpy objects is zeros_like.
  """
  assert t not in grad_initializers
  grad_initializers[t] = (init_grad_function, True)


def init_grad(obj, allow_lazy_initializer=False):
  """Initialize the gradient for an object.

  Args:
    obj: The object to initialize the gradient for, can be either a number,
      array, tuple, list, or dictionary.
    allow_lazy_initializer: Whether to allow using the ZeroGradient wrapper,
      for efficiency.

  Returns:
    An object of the same type, shape, etc. but with all numeric values set to
    zero. If the type is unknown, a zero is returned.
  """
  if obj is None:
    # TODO: fixes.py appears to pass None value and expect 0.0 back. Bug?
    return 0.0

  initializer, supports_lazy_initializer = grad_initializers[type(obj)]
  if supports_lazy_initializer:
    if isinstance(obj, ZeroGradient):
      if allow_lazy_initializer:
        return ZeroGradient(obj.like)
      else:
        # TODO: Not sure this should normally be hit. In forward-over-reverse?
        return obj.instantiate()
    else:
      if allow_lazy_initializer:
        return ZeroGradient(obj)
  else:
    assert not isinstance(obj, ZeroGradient)
  return initializer(obj)


def add_grad_numpy(left, right):
  right = unbroadcast(numpy.array(right), left)
  return left + right


def add_grad_list(left, right):
  return [add_grad(l, r) for l, r in zip(left, right)]


def add_grad_dict(left, right):
  assert all(k in left for k in right)
  return {k: add_grad(left[k], right[k]) for k in left}


grad_adders = {
    (list, list): add_grad_list,
    (dict, dict): add_grad_dict,
    (numpy.ndarray, numpy.ndarray): add_grad_numpy,
    (numpy.ndarray, float): add_grad_numpy,
    (float, numpy.ndarray): add_grad_numpy,
    (numpy.ndarray, numpy.float64): add_grad_numpy,
    (numpy.float64, numpy.ndarray): add_grad_numpy,
    (numpy.ndarray, list): add_grad_numpy,
    (list, numpy.ndarray): add_grad_numpy,
    (bool, bool): lambda left, right: left or right,
}


def register_add_grad(left_type, right_type, add_grad_function):
  """Register a new gradient adder supporting the given types.

  Gradient adders are used to add (in the sense of arithmetic addition)
  intermediate adjoint and tangent variables.
  TODO: Link to the document explaining the overall terminology and mechanics.

  Args:
    left_type: A Python type object. The data type of the left operand
      supported by the adder.
    right_type: A Python type object. The data type of the right operand
      supported by the adder.
    add_grad_function: A binary function that takes two arguments, left and
      right, of the types left_type and right_type respectively, and returns
      their sum. For example, the gradient adder for Numpy objects is np.add.
  """
  grad_adders[(left_type, right_type)] = add_grad_function


def register_all_add_grad(add_grad_function, *types):
  """Register a gradient adder for all combinations of given types.

  This is a convenience shorthand for calling register_add_grad when registering
  gradient adders for multiple types that can be interchanged for the purpose
  of addition.

  Args:
    add_grad_function: A gradient adder, see register_add_grad.
    *types: Python type objects. The gradient adder will be registered for all
      pairs of these types.
  """
  for t1 in types:
    for t2 in types:
      register_add_grad(t1, t2, add_grad_function)


register_all_add_grad(
    lambda left, right: left + right,
    float, int, numpy.int64, numpy.float32, numpy.float64)


def add_grad(left, right):
  """Recursively add the gradient of two objects.

  Args:
    left: The left value to add. Can be either an array, a number, list or
        dictionary.
    right: The right value. Must be of the same type (recursively) as the left.

  Returns:
    The sum of the two gradients, which will of the same type.
  """
  # We assume that initial gradients are always identity WRT add_grad.
  # We also assume that only init_grad could have created None values.
  assert left is not None and right is not None
  left_type = type(left)
  right_type = type(right)
  if left_type is ZeroGradient:
    return right
  if right_type is ZeroGradient:
    return left
  return grad_adders[(left_type, right_type)](left, right)


def push(stack, x, op_id):
  """Push a value onto the stack (i.e. record it on the tape).

  Args:
    stack: The stack object, which must support appending values.
    x: The value to append. If it is a mutable object like an array or list, it
        will be copied before being added onto the stack.
    op_id: A unique variable that is also passed into the corresponding pop.
        Allows optimization passes to track pairs of pushes and pops.
  """
  if isinstance(x, numpy.ndarray):
    x = x.copy()
  elif isinstance(x, list):
    x = x[:]
  if __debug__:
    stack.append((x, op_id))
  else:
    stack.append(x)


def pop(stack, op_id):
  """Pop a value from the stack (i.e. read it from the tape).

  Args:
    stack: The stack to pop from.
    op_id: A unique variable that is also passed into the matching push.
        Allows optimization passes to track pairs of pushes and pops.

  Returns:
    The last value.
  """
  if __debug__:
    pushed_value, pushed_op_id = stack.pop()
    assert pushed_op_id == op_id, 'Wanted %s, got %s' % (op_id, pushed_op_id)
  else:
    pushed_value = stack.pop()
  return pushed_value


def pop_stack(stack, op_id):
  """Proxy of pop, where we know we're popping a stack off of a stack.

  We know that we don't need to differentiate through this.
  See pop() for more.

  Args:
    stack: The stack to pop from.
    op_id: A unique variable that is also passed into the matching push.
        Allows optimization passes to track pairs of pushes and pops.

  Returns:
    The last value.
  """
  if __debug__:
    pushed_stack, pushed_op_id = stack.pop()
    assert pushed_op_id == op_id, 'Wanted %s, got %s' % (op_id, pushed_op_id)
  else:
    pushed_stack = stack.pop()
  return pushed_stack


def push_stack(stack, substack, op_id):
  """Proxy of push, where we know we're pushing a stack onto a stack.

  Used when differentiating call trees,where sub-functions get their own stack.
  See push() for more.

  Args:
    stack: The stack object, which must support appending values.
    substack: The stack to append.
    op_id: A unique variable that is also passed into the corresponding pop.
        Allows optimization passes to track pairs of pushes and pops.

  Raises:
    ValueError: If a non-stack value for `substack` is passed.
  """
  if substack is not None and not isinstance(substack, Stack):
    raise ValueError(
        'Substack should be type tangent.Stack or None, instead found %s' %
        type(substack))
  if __debug__:
    stack.append((substack, op_id))
  else:
    stack.append(substack)


non_differentiable.register_non_differentiable_functions(
    init_grad, array_size, Stack)


def insert_grad_of(var):
  """The context manager that allows insertion of arbitrary adjoint code.

  This function can be used as a context manager e.g. `with insert_grad_of(x) as dx`
  to write code that will be inserted in the adjoint while having access to the
  gradients of certain variables.

  This function is handled by reverse mode automatic differentiation, and
  shouldn't actually ever be called. If the user wants to use a function
  containing this context manager without taking the derivative, the `tangent`
  decorator should be used to remove it from the code.

  Args:
    var: The variable of which we want the gradient.

  Returns:
    The gradient of this value.

  Raises:
    ValueError: If this context manager isn't removed using the `tangent`
        decorator and the code is actually run.
  """
  raise ValueError('use the tangent decorator for functions containing '
                   'the `with insert_grad_of` statement')


def grad_dot(dy, x1, x2):
  """Gradient of NumPy dot product w.r.t. to the left hand side.

  Args:
    dy: The gradient with respect to the output.
    x1: The left hand side of the `numpy.dot` function.
    x2: The right hand side

  Returns:
    The gradient with respect to `x1` i.e. `x2.dot(dy.T)` with all the
    broadcasting involved.
  """
  if len(numpy.shape(x1)) == 1:
    dy = numpy.atleast_2d(dy)
  elif len(numpy.shape(x2)) == 1:
    dy = numpy.transpose(numpy.atleast_2d(dy))
    x2 = numpy.transpose(numpy.atleast_2d(x2))
  x2_t = numpy.transpose(numpy.atleast_2d(
      numpy.sum(x2, axis=tuple(numpy.arange(numpy.ndim(x2) - 2)))))
  dy_x2 = numpy.sum(dy, axis=tuple(-numpy.arange(numpy.ndim(x2) - 2) - 2))
  return numpy.reshape(numpy.dot(dy_x2, x2_t), numpy.shape(x1))


def copy(source):
  source_type = type(source)
  # If gradient initializers are allowed, then the object is immutable and we
  # don't need to deep copy it.
  if source_type in grad_initializers:
    _, allow_lazy_init_grad = grad_initializers[source_type]
    if allow_lazy_init_grad:
      return source
  return native_copy(source)
