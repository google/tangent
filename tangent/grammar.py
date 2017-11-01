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
"""Classifications of AST nodes."""
from __future__ import absolute_import
import gast

LITERALS = (gast.Num, gast.Str, gast.Bytes, gast.Ellipsis, gast.NameConstant)

CONTROL_FLOW = (gast.For, gast.AsyncFor, gast.While, gast.If, gast.Try)

COMPOUND_STATEMENTS = (
    gast.FunctionDef,
    gast.ClassDef,
    gast.For,
    gast.While,
    gast.If,
    gast.With,
    gast.Try,
    gast.AsyncFunctionDef,
    gast.AsyncFor,
    gast.AsyncWith
)

SIMPLE_STATEMENTS = (
    gast.Return,
    gast.Delete,
    gast.Assign,
    gast.AugAssign,
    gast.Raise,
    gast.Assert,
    gast.Import,
    gast.ImportFrom,
    gast.Global,
    gast.Nonlocal,
    gast.Expr,
    gast.Pass,
    gast.Break,
    gast.Continue
)

STATEMENTS = COMPOUND_STATEMENTS + SIMPLE_STATEMENTS

BLOCKS = (
    (gast.Module, 'body'),
    (gast.FunctionDef, 'body'),
    (gast.AsyncFunctionDef, 'body'),
    (gast.For, 'body'),
    (gast.For, 'orelse'),
    (gast.AsyncFor, 'body'),
    (gast.AsyncFor, 'orelse'),
    (gast.While, 'body'),
    (gast.While, 'orelse'),
    (gast.If, 'body'),
    (gast.If, 'orelse'),
)
