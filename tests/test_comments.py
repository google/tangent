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
import pytest

from tangent import comments
from tangent import quoting


def f(x):
  y = x
  return y


def test_comment():
  node = quoting.parse_function(f).body[0]

  comments.add_comment(node.body[0], 'foo', 'above')
  source = quoting.to_source(node)
  lines = source.split('\n')
  assert lines[1].strip() == '# foo'

  comments.add_comment(node.body[0], 'foo', 'right')
  source = quoting.to_source(node)
  lines = source.split('\n')
  assert lines[1].strip() == 'y = x # foo'

  comments.add_comment(node.body[0], 'foo', 'below')
  source = quoting.to_source(node)
  lines = source.split('\n')
  assert lines[2].strip() == '# foo'


if __name__ == '__main__':
  assert not pytest.main([__file__])
