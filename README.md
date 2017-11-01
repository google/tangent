# Tangent

## Python subset

Many of Python's advanced features are difficult to statically analyze or to
define sensible gradients of, so we restrict Python to a functional subset
(i.e. no mutable objects).

## AD

The `ad` function is a context-free pass that returns a primal and adjoint for
each node in the AST. The `joint` motion simply merges these two bodies
together, whereas the `split` motion will make sure that the state of the
primal is stored onto the stack and restored in the adjoint function.

The resulting functions are are still not runnable for two reasons:

* The initial gradient of variables is not defined i.e. we must change `dx
  += dx0` into `dx = dx0` if `dx` is not defined, or add `dx = 0` at the
  beginning of the function
* Similarly, variables being pushed to the stack might not be defined, so we
  want to either remove these pushes, or define the variable explicitly

Once these changes are made, we should have correct gradients.

## Optimization

We are often interested in the gradients of only some of the arguments. In this
case, many of the adjoint calculation might be dead code. In the optimization
pass this is removed.

## Closures

Closures are currently not supported for the following reasons:

* AD relies on being able to resolve function names. If function names are
  resolved using the enclosing function namespace, we cannot be sure that they
  will resolve to the same function at each call.
* Although we can access functions from the enclosing function namespace, we
  cannot write to this namespace, which is required for the gradients.

## Overview

Roughly the following functions are called to go from a function to ASTs for a
split or joint motion gradient.

![Overview](grad.png)
.
.
.

## Installing and running

### Installation

The easiest way to install is by cloning the git repo. Then from the root
directory, use either of the methods listed below.

#### Conda

    conda env create -f environment.yml
    conda activate tangent

#### PIP

To install Tangent and its dependencies:

    pip install -e .

To install just the requirements:

    pip install -r requirements.txt

### Running tests

Tests require pytest:

    pip install pytest

To run the tests:

    python -m pytest --short tests
