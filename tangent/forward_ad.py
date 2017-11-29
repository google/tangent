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
"""Perform forward mode automatic differentiation on an AST.

This module contains the machinery to take the AST of a function and to return
the AST of the derivative. It does so by first
walking the tree with the `ForwardAD` transformer, emitting both primal
and forward-mode statements in pairs of all statements. Non-statements AST
nodes, like conditionals, loops and expressions, are unmodified. The function
name is updated to reflect its new functionality, and its argument signature
is appended with a variable for the derivative direction.
"""
from __future__ import absolute_import
import inspect
import gast
import six
import tangent
from tangent import annotate
from tangent import annotations as anno
from tangent import ast as ast_
from tangent import cfg
from tangent import comments
from tangent import create
from tangent import errors
from tangent import funcsigs
from tangent import grads
from tangent import naming
from tangent import non_differentiable
from tangent import quoting
from tangent import tangents
from tangent import template
from tangent import tracing
from tangent import transformers
from tangent import utils


class ForwardAD(transformers.TreeTransformer):
  """Generate a primal and adjoint for a given AST tree.

  This class walks the AST recursively and for each statement (currently just
  Assign, since we assume all AugAssign nodes have been removed by ANF)
  returns a new primal and tangent statement. Note that it relies on function
  calls being resolved by the `resolve_call` function.

  Args:
    wrt: A tuple of argument indices with respect to which the gradient should
        be taken.
    preserve_result: A boolean indicating whether the return value
        of the original function should be preserved. If True, will append
        the original return value to the derivative in a tuple.
    check_dims: A boolean indicating whether the user-provided derivatives
        must have the same shape as their corresponding arguments. For example,

        > f = lambda x: x * x
        > df = autodiff(f,mode='forward',check_dims=True)
        > df(3.0, 1.0) # succeeds
        > df(np.eye(3), 1.0) # fails, x is a matrix, dx is a scalar


  Attributes:
    required: List of user-defined functions that the primal calls.
        The primals and adjoints of this function need to be available in the
        global namespace.
  """

  def __init__(self, wrt, preserve_result, check_dims):
    self.wrt = wrt
    self.required = []
    self.target = None
    self.metastack = []
    self.preserve_result = preserve_result
    self.check_dims = check_dims
    super(ForwardAD, self).__init__()
    self._tmp_node = None

  @property
  def tmp_node(self):
    if self._tmp_node is None:
      self._tmp_node = quoting.quote(self.namer.unique('tmp'))
    return self._tmp_node

  def reset_tmp_node(self):
    self._tmp_node = None

  def visit(self, node):
    method = 'visit_' + node.__class__.__name__
    visitor = getattr(self, method, self.generic_visit)

    # Set certain attributes for child nodes
    if anno.hasanno(node, 'active_in'):
      self.active_variables = anno.getanno(node, 'active_in')

    return visitor(node)

  def is_active(self, node):
    active_variables = anno.getanno(node, 'active_in')
    for succ in gast.walk(node):
      if (isinstance(succ, gast.Name) and isinstance(succ.ctx, gast.Load) and
          succ.id in active_variables):
        return True
    return False

  def visit_FunctionDef(self, node):
    self.namer = naming.Namer.build(node)

    # Get the tangent of the body
    new_body = []
    for n in node.body:
      new = self.visit(n)
      if isinstance(new, (list, tuple)):
        new_body.extend(new)
      else:
        new_body.append(new)
    node.body = new_body

    # Add in the initial gradient argument
    grad_args = [
        create.create_grad(arg, self.namer, tangent=True)
        for i, arg in enumerate(node.args.args) if i in self.wrt
    ]
    if len(self.wrt) != len(grad_args):
      raise ValueError(
          'Mismatch between requested and retrieved derivative arguments. '
          'Requested %d, found %d') % (len(self.wrt), len(grad_args))

    node.args.args += grad_args

    if self.check_dims:
      # Define the shape check code quote
      def shape_match_template(primal, tangent_):
        if not tangent.shapes_match(primal, tangent_):
          raise ValueError(
              'Shape mismatch between argument value (%s) and seed derivative '
              '(%s)' \
        % (numpy.shape(primal), numpy.shape(tangent_)))

      # Add a shape check for each seed derivative & primal pair.
      shape_check_nodes = []
      for iwrt, tangent_var in zip(self.wrt, grad_args):
        primal = node.args.args[iwrt]
        shape_check = template.replace(
            shape_match_template, primal=primal, tangent_=tangent_var)[0]
        shape_check_nodes.append(shape_check)
      node.body = shape_check_nodes + node.body

    # Add in gradient initialization statements for everything else
    grad_init_nodes = [
        template.replace(
            'd[z] = init_grad(z)',
            replace_grad=template.Replace.TANGENT,
            namer=self.namer,
            z=arg,
            init_grad=utils.INIT_GRAD) for i, arg in enumerate(node.args.args)
        if i not in self.wrt
    ]
    node.body = grad_init_nodes + node.body

    # Rename the function
    func = anno.getanno(node, 'func')
    node.name = naming.tangent_name(func, self.wrt)

    return node

  def visit_Assign(self, node):
    """Visit assignment statement.

    Notes
    -----
    This method sets the `self.target` attribute to the first assignment
    target for callees to use.
    """

    # Generate tangent name of the assignment
    self.target = node.targets[0]
    self.value = node.value

    # Get the tangent node
    tangent_node = self.visit(self.value)

    # If no forward-mode statement was created, then no extra work is needed.
    if tangent_node == self.value:
      self.target = None
      return node

    if self.value:
      new_node = template.replace(
          tangents.tangents[gast.Assign],
          replace_grad=template.Replace.TANGENT,
          namer=self.namer,
          temp=self.tmp_node,
          tangent=tangent_node,
          target=self.target,
          value=self.value)
      # Ensure that we use a unique tmp node per primal/tangent pair
      self.reset_tmp_node()
    else:
      # We're already in ANF form right now,
      # so we can assume LHS is a single Name "z"
      def template_(z, f):
        tmp = f
        d[z] = tmp[0]
        z = tmp[1]

      new_node = template.replace(
          template_,
          replace_grad=template.Replace.TANGENT,
          namer=self.namer,
          z=self.target,
          f=tangent_node[0])

    # Add it after the original statement
    self.target = None

    # Add in some cool comments
    for i in range(len(new_node)):
      new_node[i] = comments.add_comment(new_node[i], 'Primal and tangent of: '
                                         '%s' % quoting.unquote(node))
    return new_node

  def visit_Call(self, node):
    if not self.target:
      return node
    func = anno.getanno(node, 'func')

    if func in tangents.UNIMPLEMENTED_TANGENTS:
      raise errors.ForwardNotImplementedError(func)

    if func == tracing.Traceable:
      raise NotImplementedError('Tracing of %s is not enabled in forward mode' %
                                quoting.unquote(node))

    if func not in tangents.tangents:
      try:
        quoting.parse_function(func)
      except:
        raise ValueError('No tangent found for %s, and could not get source.' %
                         func.__name__)

      # z = f(x,y) -> d[z],z = df(x,y,dx=dx,dy=dy)
      active_args = tuple(i for i, arg in enumerate(node.args)
                          if isinstance(arg, gast.Name))
      # TODO: Stack arguments are currently not considered
      # active, but for forward-mode applied to call trees,
      # they have to be. When we figure out how to update activity
      # analysis to do the right thing, we'll want to add the extra check:
      # `and arg.id in self.active_variables`

      # TODO: Duplicate of code in reverse_ad.
      already_counted = False
      for f, a in self.required:
        if f.__name__ == func.__name__ and set(a) == set(active_args):
          already_counted = True
          break
      if not already_counted:
        self.required.append((func, active_args))

      fn_name = naming.tangent_name(func, active_args)
      orig_args = quoting.parse_function(func).body[0].args
      tangent_keywords = []
      for i in active_args:
        grad_node = create.create_grad(node.args[i], self.namer, tangent=True)
        arg_grad_node = create.create_grad(
            orig_args.args[i], self.namer, tangent=True)
        grad_node.ctx = gast.Load()
        tangent_keywords.append(
            gast.keyword(arg=arg_grad_node.id, value=grad_node))
      # Update the original call
      rhs = gast.Call(
          func=gast.Name(id=fn_name, ctx=gast.Load(), annotation=None),
          args=node.args,
          keywords=tangent_keywords + node.keywords)
      # Set self.value to False to trigger whole primal replacement
      self.value = False
      return [rhs]

    template_ = tangents.tangents[func]

    # Match the function call to the template
    sig = funcsigs.signature(template_)
    sig = sig.replace(parameters=list(sig.parameters.values())[1:])
    kwargs = dict((keyword.arg, keyword.value) for keyword in node.keywords)
    bound_args = sig.bind(*node.args, **kwargs)

    # Fill in any missing kwargs with the defaults from the template
    args = quoting.parse_function(template_).body[0].args
    kwargs = dict(zip(*map(reversed, [args.args, args.defaults])))
    kwargs.update(dict(zip(args.kwonlyargs, args.kw_defaults)))
    for arg, val in kwargs.items():
      if arg.id not in bound_args.arguments:
        bound_args.arguments[arg.id] = val

    # Let's fill in the template. The first argument is the output, which
    # was stored in a temporary variable
    output_name = six.get_function_code(template_).co_varnames[0]
    arg_replacements = {output_name: self.tmp_node}
    arg_replacements.update(bound_args.arguments)

    # If the template uses *args, then we pack the corresponding inputs
    flags = six.get_function_code(template_).co_flags

    if flags & inspect.CO_VARARGS:
      to_pack = node.args[six.get_function_code(template_).co_argcount - 1:]
      vararg_name = six.get_function_code(template_).co_varnames[-1]
      target = gast.Name(annotation=None, id=vararg_name, ctx=gast.Store())
      value = gast.Tuple(elts=to_pack, ctx=gast.Load())

      # And we fill in the packed tuple into the template
      arg_replacements[six.get_function_code(template_).co_varnames[
          -1]] = target
    tangent_node = template.replace(
        template_,
        replace_grad=template.Replace.TANGENT,
        namer=self.namer,
        **arg_replacements)

    # If the template uses the answer in the RHS of the tangent,
    # we need to make sure that the regular answer is replaced
    # with self.tmp_node, but that the gradient is not. We have
    # to be extra careful for statements like a = exp(a), because
    # both the target and RHS variables have the same name.
    tmp_grad_node = create.create_grad(self.tmp_node, self.namer, tangent=True)
    tmp_grad_name = tmp_grad_node.id
    ans_grad_node = create.create_grad(self.target, self.namer, tangent=True)
    for _node in tangent_node:
      for succ in gast.walk(_node):
        if isinstance(succ, gast.Name) and succ.id == tmp_grad_name:
          succ.id = ans_grad_node.id

    if flags & inspect.CO_VARARGS:
      # If the template packs arguments, then we have to unpack the
      # derivatives afterwards
      # We also have to update the replacements tuple then
      dto_pack = [
          create.create_temp_grad(arg, self.namer, True) for arg in to_pack
      ]
      value = create.create_grad(target, self.namer, tangent=True)
      target = gast.Tuple(elts=dto_pack, ctx=gast.Store())

    # Stack pops have to be special-cased, we have
    # to set the 'push' attribute, so we know that if we
    # remove this pop, we have to remove the equivalent push.
    # NOTE: this only works if we're doing forward-over-reverse,
    # where reverse is applied in joint mode, with no call tree.
    # Otherwise, the pushes and pops won't be matched within a single
    # function call.
    if func == tangent.pop:
      if len(self.metastack):
        anno.setanno(tangent_node[0], 'push', self.metastack.pop())
      else:
        anno.setanno(tangent_node[0], 'push', None)
    return tangent_node

  def visit_BinOp(self, node):

    if not self.target:
      return node

    template_ = tangents.tangents[node.op.__class__]
    tangent_node = template.replace(
        template=template_,
        replace_grad=template.Replace.TANGENT,
        namer=self.namer,
        x=node.left,
        y=node.right,
        z=self.target)
    return tangent_node

  def visit_UnaryOp(self, node):
    if not self.target:
      return node

    template_ = tangents.tangents[node.op.__class__]
    tangent_node = template.replace(
        template=template_,
        replace_grad=template.Replace.TANGENT,
        namer=self.namer,
        x=node.operand,
        z=self.target)
    return tangent_node

  def visit_Expr(self, node):
    # Special-case the push() expression (have to reverse the usual
    # tangent/primal order)

    if isinstance(node.value, gast.Call):
      fn = anno.getanno(node.value, 'func')
      if fn in [tangent.push, tangent.push_stack]:
        # Save the pop associated with this push
        template_ = tangents.tangents[fn]
        stack_node, var_node, op_id = node.value.args
        tangent_node = template.replace(
            template_,
            replace_grad=template.Replace.TANGENT,
            namer=self.namer,
            x=var_node,
            stack=stack_node,
            op_id=op_id)
        # Push the original node and the tangent_node push
        # onto a "meta-stack", so we can track the relationship
        # between the pushes and pops in tangent mode
        self.metastack.append(tangent_node[0])
        return [node] + tangent_node
    return node

  def visit_Num(self, node):
    """Tangent of e.g.
    x = 0"""
    if not self.target:
      return node

    template_ = tangents.tangents[node.__class__]
    tangent_node = template.replace(
        template_,
        replace_grad=template.Replace.TANGENT,
        namer=self.namer,
        x=node,
        z=self.target)
    return tangent_node

  def visit_Name(self, node):
    """Tangent of e.g.
    x = y"""
    if not self.target:
      return node

    # The gast representation of 'None' is as a name in some version fo Python,
    # not as a special NameConstant. So, we have to do a bit of
    # special-casing here. Ideally, this is fixed in gast.
    if node.id in ['None','True','False']:
      template_ = 'd[z] = x'
    else:
      template_ = tangents.tangents[node.__class__]
    tangent_node = template.replace(
        template=template_,
        replace_grad=template.Replace.TANGENT,
        namer=self.namer,
        x=node,
        z=self.target)
    return tangent_node

  def visit_NameConstant(self, node):
    """Tangent of e.g.
    x = None

    Lines of this type are sometimes auto-generated by reverse-mode,
    and thus handling them is important for higher-order autodiff
    We will shunt NameConstant tangents off to visit_Name, to prevent
    code duplication.
    """
    constant_val = {
        True: 'True',
        False: 'False',
        None: 'None',
    }[node.value]
    new_node = gast.Name(id=constant_val,ctx=gast.Load(),annotation=None)
    return self.visit_Name(new_node)

  def visit_Attribute(self, node):
    """Tangent of e.g.
    x = y.z"""
    if not self.target:
      return node

    template_ = tangents.tangents[node.__class__]
    tangent_node = template.replace(
        template=template_,
        replace_grad=template.Replace.TANGENT,
        namer=self.namer,
        x=node,
        z=self.target)
    return tangent_node

  def visit_Subscript(self, node):
    """Tangent of e.g.
    x = y[i]"""
    if not self.target:
      return node

    template_ = tangents.tangents[node.__class__]
    tangent_node = template.replace(
        template=template_,
        replace_grad=template.Replace.TANGENT,
        namer=self.namer,
        x=node,
        z=self.target)
    return tangent_node

  def create_grad_list(self, node):
    assert isinstance(node, (gast.List, gast.Tuple)), 'Must be list or tuple'
    list_of_nodes = node.elts
    elts = []
    for _node in list_of_nodes:
      if isinstance(_node, (gast.Name, gast.Subscript)):
        grad_node = create.create_grad(_node, self.namer, tangent=True)
        grad_node.ctx = node.ctx
        elts.append(grad_node)
      elif isinstance(_node, gast.Num):
        elts.append(gast.Num(0))
      elif isinstance(_node, (gast.List, gast.Tuple)):
        elts.append(self.create_grad_list(_node.elts))
      else:
        raise ValueError('Cannot handle node type %s' % type(_node))

    return node.__class__(elts=elts, ctx=node.ctx)

  def visit_List(self, node):
    if not self.target:
      return node
    dlist = self.create_grad_list(node)
    tangent_node = [
        template.replace(
            'd[z] = dlist',
            replace_grad=template.Replace.TANGENT,
            namer=self.namer,
            z=self.target,
            dlist=dlist)
    ]
    return tangent_node

  def visit_Tuple(self, node):
    return self.visit_List(node)

  def visit_Return(self, node):
    orig_retval = ast_.copy_node(node.value)
    retval = node.value
    if isinstance(retval, (gast.Name, gast.Subscript)):
      retval = gast.Tuple(
          elts=[create.create_grad(retval, self.namer, tangent=True)],
          ctx=gast.Load())
    elif isinstance(retval, gast.Tuple):
      retval.elts = [
          create.create_grad(elt, self.namer, tangent=True)
          for elt in retval.elts
      ]
    else:
      raise ValueError
    for n in retval.elts:
      n.ctx = gast.Load()
    if self.preserve_result:
      retval.elts.append(orig_retval)
    node.value = retval
    return node


def forward_ad(node, wrt, preserve_result=False, check_dims=True):
  """Perform forward-mode AD on an AST.

  This function analyses the AST to determine which variables are active and
  proceeds by taking the naive derivative. Before returning the primal and
  adjoint it annotates push and pop statements as such.

  Args:
    node: A `FunctionDef` AST node.
    wrt: A tuple of argument indices with respect to which we take the
        derivative.
    preserve_result: A boolean indicating whether the original
        non-differentiated function value should be returned
    check_dims: A boolean indicating whether the provided derivatives should
        have the same shape as their corresponding arguments.

  Returns:
    mod: A `Module` node containing the naive primal and adjoint of the
        function which can be fed to the `split` and `joint` functions.
    required: A list of tuples of functions and argument indices. These
        functions were called by the function but did not have an adjoint.
  """
  if not isinstance(node, gast.FunctionDef):
    raise TypeError

  # Activity analysis
  cfg_obj = cfg.CFG.build_cfg(node)
  cfg.Active(range(len(node.args.args))).visit(cfg_obj.entry)

  # Build forward mode function
  fad = ForwardAD(wrt, preserve_result, check_dims)
  node = fad.visit(node)

  # Annotate stacks
  node = annotate.find_stacks(node)

  # Clean up naive forward-mode fcode
  node = gast.Module([node])
  anno.clearanno(node)

  return node, fad.required
