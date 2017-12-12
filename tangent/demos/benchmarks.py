"""Performance tests.

TODO: Add synopsis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import cProfile
import os
import pstats
import signal

try:
  from StringIO import StringIO
except ImportError:
  from io import StringIO

import sys
import time

from autograd import grad
import numpy as np

from tangent.demos import basic_models
from tangent.demos import models
from tangent.demos import train_utils
from tangent.demos.common_kernels import softmax_crossent

if os.environ.get('USES_TORCH', False):
  import torch
  from torchvision.models import resnet50
else:
  # TODO: What exactly is the source of segfault? Doesn't seem to be tf.
  from tangent.grad_util import grad
  from tensorflow.contrib.eager.python import tfe

import tensorflow as tf


def _tag_prefix():
  return os.environ.get('RUN_NUMBER', '')


def _target_device():
  if os.environ.get('USES_TORCH', False):
    return ''
  if tfe.num_gpus() > 0 and os.environ.get('TF_USES_CPU', 'false') != 'true':
    return '/gpu'
  return '/cpu'


def _have_gpu():
  return 'gpu' in _target_device()


def _array_side_sizes():
  return (64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _train_batch_sizes():
  if _have_gpu():
    return (16, 32)
  return (16, 32)


def _num_iterations(batch_size, train):
  num_iters = min(30, 2000 // batch_size)
  if not _have_gpu():
    num_iters //= 3
  if train:
    num_iters //= 3
  return max(3, num_iters)


def _outputs_file_path(file_name):
  directory = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR')
  if directory is None:
    directory = '/tmp/'
  return os.path.join(directory, file_name)


def _cprofile(tag):
  return profile(dump_to_file=_outputs_file_path('%s.cprof' % tag))


def _sampling_profile(tag):
  return sampling_profile(
      filename=_outputs_file_path('%s.flame' % tag))


@contextlib.contextmanager
def profile(order_by='tottime', max_output=1024, dump_to_file=None):
  """Profile the time of each python function.

    Simple wrapper of https://docs.python.org/2/library/profile.html
    User can visualize it by
      set dump_to_file='/tmp/script.profile'
      pyprof2calltree -k -i /tmp/script.profile

  Args:
    order_by: The attribute to order the result.
    max_output: Max length of stdout output.
    dump_to_file: File to dump the profile to.
  Yields:
    Nothing.
  """
  pr = cProfile.Profile()
  pr.enable()
  yield
  pr.disable()
  s = StringIO()
  ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(order_by)
  ps.print_stats()

  if max_output:
    stat_val = s.getvalue()
    out_size = min(max_output, len(stat_val))
    sys.stdout.write('%s\n' % stat_val[:out_size])
    sys.stdout.flush()
  if dump_to_file:
    pr.dump_stats(dump_to_file)


class Timer(object):
  """A Timer that tracks the average time of n steps.

  Args:
    every_n_steps: Show the average time of every this steps.
  """

  def __init__(self, every_n_steps=1, name='timer'):
    self._every_n_steps = every_n_steps
    self._count = 0
    self._start_clock = 0
    self._start_time = 0
    self._name = name

  def start(self):
    if self._count == 0:
      self._start_clock = time.clock()
      self._start_time = time.time()

  def end(self):
    self._count += 1
    if self._count == self._every_n_steps:
      clock_diff = time.clock() - self._start_clock
      time_diff = time.time() - self._start_time
      sys.stdout.write('%s: %.2f process seconds/step, %.2f wall secs/step.\n' %
                       (self._name,
                        1.0 * clock_diff / self._every_n_steps,
                        1.0 * time_diff / self._every_n_steps))
      sys.stdout.flush()
      self._count = 0


class Sampler(object):
  """A simple signal-based stack sampling profiler.

  More accurate than the tracing profiler when it comes to representing the time
  function X takes when called by function Y as opposed to the total time taken
  by function X (which is the only information in the tracing profiler).
  """

  def __init__(self, interval=None):
    self.stack_counts = collections.defaultdict(int)
    self.interval = 0.001 if interval is None else interval

  def _sample(self, signum, frame):
    """Samples the current stack."""
    del signum
    stack = []
    while frame is not None:
      formatted_frame = '{}({})'.format(frame.f_code.co_name,
                                        frame.f_globals.get('__name__'))
      stack.append(formatted_frame)
      frame = frame.f_back

    formatted_stack = ';'.join(reversed(stack))
    self.stack_counts[formatted_stack] += 1
    signal.setitimer(signal.ITIMER_VIRTUAL, self.interval, 0)

  @contextlib.contextmanager
  def profile(self):
    signal.signal(signal.SIGVTALRM, self._sample)
    signal.setitimer(signal.ITIMER_VIRTUAL, self.interval, 0)
    try:
      yield
    finally:
      signal.setitimer(signal.ITIMER_VIRTUAL, 0)

  def save(self, fname):
    stacks = self.stack_counts
    with open(fname, 'w') as f:
      for s in stacks:
        f.write('%s %s\n' % (s, stacks[s]))


@contextlib.contextmanager
def sampling_profile(filename, interval=None):
  """Sampling-based profiler.

  To use, pass a filename and then run

  $ perl FlameGraph/flamegraph.pl  filename  > filename.svg

  then open the svg file in the browser.

  Args:
    filename: the filename
    interval: the sampling interval, in seconds. Defaults to 0.001.

  Yields:
   nothing
  """
  sampler = Sampler(interval=interval)
  with sampler.profile():
    yield
  sampler.save(filename)


class _ReportingBenchmark(tf.test.Benchmark):

  def report_results(self, tag, start, num_iters, batch_size):
    avg_time = (time.time() - start) / num_iters
    extras = {'examples_per_sec': batch_size / avg_time}
    self.report_benchmark(
        iters=num_iters, wall_time=avg_time, name=tag, extras=extras)

  def create_random_values(self, batch_size, shape):
    shape = (batch_size,) + shape
    return np.random.uniform(size=shape).astype(np.float32)

  def create_random_categories(self, batch_size, num_labels, one_hot):
    np_label = np.random.uniform(
        size=(batch_size,), low=0, high=num_labels).astype(np.int64)
    if not one_hot:
      return np_label
    np_one_hot_label = np.zeros((batch_size, num_labels)).astype(np.float32)
    np_one_hot_label[np.arange(batch_size), np_label] = 1
    return np_one_hot_label


class SimpleModelsBenchmark(_ReportingBenchmark):

  def _create_inputs(self, batch_size, input_size):
    return self.create_random_values(batch_size, (input_size,))

  def _create_labels(self, batch_size):
    return self.create_random_values(batch_size, ())

  def _tag(self, name, batch_size, array_side_size):
    dev = 'gpu' if _have_gpu() else 'cpu'
    tag = '%s.%s.batch_%d.size_%d' % (name, dev, batch_size, array_side_size)
    if _tag_prefix():
      tag = '%s_%s' % (_tag_prefix(), tag)
    return tag

  def _trainer_mlp_numpy_autograd(self, num_features, input_size):
    w1 = np.random.normal(
        size=(input_size, num_features),
        scale=1.0 / (num_features * input_size)).astype(np.float32)
    b1 = np.random.normal(size=(num_features,)).astype(np.float32)
    wout = np.random.normal(size=(num_features, 1)).astype(np.float32)
    bout = np.random.normal(size=(1,)).astype(np.float32)

    def loss(x, w1, b1, wout, bout, label):
      return basic_models.mlp_numpy(x, w1, b1, wout, bout, label)

    dloss = grad(loss, tuple(range(1, 5)))

    def train_one_step(inputs, labels):
      dloss(inputs, w1, b1, wout, bout, labels)
    return train_one_step

  def _trainer_rnn_numpy_autograd(self, num_features, input_size, num_steps=10):
    w1 = np.random.normal(size=(input_size, num_features)).astype(np.float32)
    b1 = np.random.normal(size=(num_features,)).astype(np.float32)
    wout = np.random.normal(size=(num_features, 1)).astype(np.float32)
    bout = np.random.normal(size=(1,)).astype(np.float32)

    def loss(x, w1, b1, wout, bout, label, num_steps):
      return basic_models.rnn_numpy(x, w1, b1, wout, bout, label, num_steps)

    dloss = grad(loss, range(1, 5))

    def train_one_step(inputs, labels):
      dloss(inputs, w1, b1, wout, bout, labels, num_steps)
    return train_one_step

  def _simple_loop_numpy_autograd(
      self, num_features, batch_size, num_steps=100):
    x = np.random.normal(size=(batch_size, num_features)).astype(np.float32)
    def loop_fn(x, num_steps):
      return basic_models.simple_loop_numpy(x, num_steps)

    dx = grad(loop_fn, (0,))

    def one_step():
      dx(x, num_steps)
    return one_step

  def _trainer_mlp_numpy_tangent(self, num_features, input_size):
    w1 = np.random.normal(
        size=(input_size, num_features),
        scale=1.0 / (num_features * input_size)).astype(np.float32)
    b1 = np.random.normal(size=(num_features,)).astype(np.float32)
    wout = np.random.normal(size=(num_features, 1)).astype(np.float32)
    bout = np.random.normal(size=(1,)).astype(np.float32)

    def loss(x, w1, b1, wout, bout, label):
      return basic_models.mlp_numpy(x, w1, b1, wout, bout, label)

    dloss = grad(loss, wrt=range(1, 5))

    def train_one_step(inputs, labels):
      dloss(inputs, w1, b1, wout, bout, labels)
    return train_one_step

  def _trainer_rnn_numpy_tangent(self, num_features, input_size, num_steps=10):
    w1 = np.random.normal(size=(input_size, num_features)).astype(np.float32)
    b1 = np.random.normal(size=(num_features,)).astype(np.float32)
    wout = np.random.normal(size=(num_features, 1)).astype(np.float32)
    bout = np.random.normal(size=(1,)).astype(np.float32)

    def loss(x, w1, b1, wout, bout, label, num_steps):
      return basic_models.rnn_numpy(x, w1, b1, wout, bout, label, num_steps)

    dloss = grad(loss, wrt=range(1, 5))

    def train_one_step(inputs, labels):
      dloss(inputs, w1, b1, wout, bout, labels, num_steps)
    return train_one_step

  def _simple_loop_numpy_tangent(
      self, num_features, batch_size, num_steps=100):
    x = np.random.normal(size=(batch_size, num_features)).astype(np.float32)
    def loop_fn(x, num_steps):
      return basic_models.simple_loop_numpy(x, num_steps)
    dx = grad(loop_fn, wrt=(0,))
    def one_step():
      dx(x, num_steps)
    return one_step

  def _trainer_mlp_pytorch(self, num_features, input_size):
    w1 = torch.autograd.Variable(
        torch.normal(
            means=torch.zeros(input_size, num_features),
            std=1.0 / (num_features * input_size)
        ).type(torch.FloatTensor),
        requires_grad=True)
    b1 = torch.autograd.Variable(
        torch.randn(num_features).type(torch.FloatTensor),
        requires_grad=True)
    wout = torch.autograd.Variable(
        torch.randn(num_features, 1).type(torch.FloatTensor),
        requires_grad=True)
    bout = torch.autograd.Variable(
        torch.randn(1).type(torch.FloatTensor),
        requires_grad=True)

    def loss(x, w1, b1, wout, bout, label):
      # TODO: Double check that this is not inefficient.
      return basic_models.mlp_pytorch(
          torch.autograd.Variable(torch.from_numpy(x)),
          w1,
          b1,
          wout,
          bout,
          torch.autograd.Variable(torch.from_numpy(label.astype(np.int64))))

    def train_one_step(inputs, labels):
      loss(inputs, w1, b1, wout, bout, labels).backward()
    return train_one_step

  def _simple_loop_pytorch(
      self, num_features, batch_size, num_steps=100):
    x = torch.autograd.Variable(
        torch.randn(batch_size, num_features).type(torch.FloatTensor),
        requires_grad=True)
    def loop_fn(x, num_steps):
      return basic_models.simple_loop_pytorch(x, num_steps)
    def one_step():
      loop_fn(x, num_steps).backward()
    return one_step

  def _trainer_mlp_tfe_tangent(self, num_features, input_size):
    w1 = tf.random_normal(
        shape=(input_size, num_features),
        stddev=1.0 / (num_features + input_size),
        dtype=tf.float32)
    b1 = tf.random_normal(shape=(num_features,), dtype=tf.float32)
    wout = tf.random_normal(shape=(num_features, 1), dtype=tf.float32)
    bout = tf.random_normal(shape=(1,), dtype=tf.float32)

    def loss(x, w1, b1, wout, bout, label):
      return basic_models.mlp_tf(x, w1, b1, wout, bout, label)

    dloss = grad(loss, wrt=range(1, 5))

    def train_one_step(inputs, labels):
      dloss(tf.constant(inputs), w1, b1, wout, bout, tf.constant(labels))
    return train_one_step

  def _trainer_rnn_tfe_tangent(self, num_features, input_size, num_steps=10):
    w1 = tf.random_normal(shape=(input_size, num_features), dtype=tf.float32)
    b1 = tf.random_normal(shape=(num_features,), dtype=tf.float32)
    wout = tf.random_normal(shape=(num_features, 1), dtype=tf.float32)
    bout = tf.random_normal(shape=(1,), dtype=tf.float32)

    def loss(x, w1, b1, wout, bout, label, num_steps):
      return basic_models.rnn_tf(x, w1, b1, wout, bout, label, num_steps)

    dloss = grad(loss, wrt=range(1, 5))

    def train_one_step(inputs, labels):
      dloss(tf.constant(inputs), w1, b1, wout, bout, tf.constant(labels),
            num_steps)
    return train_one_step

  def _simple_loop_tfe_tangent(self, num_features, batch_size, num_steps=100):
    x = tf.random_normal(shape=(batch_size, num_features), dtype=tf.float32)
    def loop_fn(x, num_steps):
      return basic_models.simple_loop_tf(x, num_steps)
    dx = grad(loop_fn, wrt=(0,))
    def one_step():
      dx(x, num_steps)
    return one_step

  def _trainer_mlp_tfe_builtin(self, num_features, input_size):
    w1 = tf.random_normal(
        shape=(input_size, num_features),
        stddev=1.0 / (num_features + input_size),
        dtype=tf.float32)
    b1 = tf.random_normal(shape=(num_features,), dtype=tf.float32)
    wout = tf.random_normal(shape=(num_features, 1), dtype=tf.float32)
    bout = tf.random_normal(shape=(1,), dtype=tf.float32)

    def loss(x, w1, b1, wout, bout, label):
      return basic_models.mlp_tf(x, w1, b1, wout, bout, label)

    dloss = tfe.gradients_function(loss, params=range(1, 5))

    def train_one_step(inputs, labels):
      dloss(inputs, w1, b1, wout, bout, labels)
    return train_one_step

  def _trainer_rnn_tfe_builtin(self, num_features, input_size, num_steps=10):
    w1 = tf.random_normal(shape=(input_size, num_features), dtype=tf.float32)
    b1 = tf.random_normal(shape=(num_features,), dtype=tf.float32)
    wout = tf.random_normal(shape=(num_features, 1), dtype=tf.float32)
    bout = tf.random_normal(shape=(1,), dtype=tf.float32)

    def loss(x, w1, b1, wout, bout, label, num_steps):
      return basic_models.rnn_tf(x, w1, b1, wout, bout, label, num_steps)

    dloss = tfe.gradients_function(loss, params=range(1, 5))

    def train_one_step(inputs, labels):
      dloss(inputs, w1, b1, wout, bout, labels, num_steps)
    return train_one_step

  def _simple_loop_tfe_builtin(self, num_features, batch_size, num_steps=100):
    x = tf.random_normal(shape=(batch_size, num_features), dtype=tf.float32)
    def loop_fn(x, num_steps):
      return basic_models.simple_loop_tf(x, num_steps)
    dx = tfe.gradients_function(loop_fn, params=(0,))
    def one_step():
      dx(x, num_steps)
    return one_step

  def _trainer_mlp_tf_builtin(self, num_features, input_size, x, label):
    w1 = tf.get_variable(
        'w1', shape=(input_size, num_features), dtype=tf.float32)
    b1 = tf.get_variable(
        'b1', shape=(num_features,), dtype=tf.float32)
    wout = tf.get_variable(
        'wout', shape=(num_features, 1), dtype=tf.float32)
    bout = tf.get_variable(
        'bout', shape=(1,), dtype=tf.float32)

    loss_tensor = basic_models.mlp_tf(x, w1, b1, wout, bout, label)
    grads = tf.gradients(loss_tensor, [w1, b1, wout, bout])
    def train_one_step(sess):
      sess.run(grads)
    return train_one_step

  def _simple_loop_tf_builtin(self, num_features, batch_size, num_steps=100):
    x = tf.get_variable('x', shape=(batch_size, num_features), dtype=tf.float32)
    loop_tensor = basic_models.simple_loop_tf_pure(x, num_steps)
    dx = tf.gradients(loop_tensor, x)
    def one_step(sess):
      sess.run(dx)
    return one_step

  def _benchmark_train(self, name, trainer_factory):
    for batch_size in _train_batch_sizes():
      for array_side_size in _array_side_sizes():
        with tf.Graph().as_default(), tf.device(_target_device()):
          try:
            tag = self._tag(name, batch_size, array_side_size)
            num_burn = 5
            num_iters = _num_iterations(batch_size, train=True)
            inputs = self._create_inputs(batch_size, array_side_size)
            labels = self._create_labels(batch_size)

            train_one_step = trainer_factory(array_side_size, array_side_size)

            for _ in range(num_burn):
              train_one_step(inputs, labels)

            start = time.time()
            for _ in range(num_iters):
              train_one_step(inputs, labels)
            self.report_results(tag, start, num_iters, batch_size)
          except:
            print('Warning: %s benchmark failed at %s with %s ' %(
                tag, batch_size, sys.exc_info()[0]))

  def _benchmark_train_graph(self, name, trainer_factory):
    for batch_size in _train_batch_sizes():
      for array_side_size in _array_side_sizes():
        with tf.Graph().as_default(), tf.device(_target_device()):
          tag = self._tag(name, batch_size, array_side_size)
          num_burn = 5
          num_iters = _num_iterations(batch_size, train=True)
          inputs = self._create_inputs(batch_size, array_side_size)
          labels = self._create_labels(batch_size)

          train_one_step = trainer_factory(
              array_side_size, array_side_size,
              tf.constant(inputs), tf.constant(labels))

          config = tf.ConfigProto(
              intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
          with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(num_burn):
              train_one_step(sess)

            start = time.time()
            for _ in range(num_iters):
              train_one_step(sess)
            self.report_results(tag, start, num_iters, batch_size)

  def _benchmark_simple(self, name, step_factory):
    for batch_size in _train_batch_sizes():
      for array_side_size in _array_side_sizes():
        with tf.Graph().as_default(), tf.device(_target_device()):
          tag = self._tag(name, batch_size, array_side_size)
          num_burn = 5
          num_iters = _num_iterations(batch_size, train=True)

          one_step = step_factory(array_side_size, batch_size)

          for _ in range(num_burn):
            one_step()

          start = time.time()
          for _ in range(num_iters):
            one_step()
          self.report_results(tag, start, num_iters, batch_size)

  def _benchmark_simple_graph(self, name, step_factory):
    for batch_size in _train_batch_sizes():
      for array_side_size in _array_side_sizes():
        with tf.Graph().as_default(), tf.device(_target_device()):
          tag = self._tag(name, batch_size, array_side_size)
          num_burn = 5
          num_iters = _num_iterations(batch_size, train=True)

          one_step = step_factory(array_side_size, batch_size)

          config = tf.ConfigProto(
              intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
          with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(num_burn):
              one_step(sess)

            start = time.time()
            for _ in range(num_iters):
              one_step(sess)
            self.report_results(tag, start, num_iters, batch_size)

  def benchmark_nips_ws_numpy(self):
    """Benchmarks for the NIPS workshop paper."""
    assert not _have_gpu()
    for i in range(int(os.environ.get('NUM_RUNS', 53))):
      self._benchmark_train(
          '%03d_nips_mlp.numpy_autograd' % i, self._trainer_mlp_numpy_autograd)
      self._benchmark_train(
          '%03d_nips_mlp.numpy_tangent' % i, self._trainer_mlp_numpy_tangent)
      self._benchmark_simple(
          '%03d_nips_loop.numpy_autograd' % i, self._simple_loop_numpy_autograd)
      self._benchmark_simple(
          '%03d_nips_loop.numpy_tangent' % i, self._simple_loop_numpy_tangent)

  def benchmark_nips_ws_graph(self):
    """Benchmarks for the NIPS workshop paper.

    Run these with --test_env=TF_USES_GRAPH_MODE=True
    """
    assert not _have_gpu()
    for i in range(int(os.environ.get('NUM_RUNS', 53))):
      self._benchmark_train_graph(
          '%03d_nips_mlp.tf_graph' % i, self._trainer_mlp_tf_builtin)
      self._benchmark_simple_graph(
          '%03d_nips_loop.tf_graph' % i, self._simple_loop_tf_builtin)

  def benchmark_nips_ws_pytorch(self):
    """Benchmarks for the NIPS workshop paper.

    Run these with --test_env=USES_TORCH=True
    """
    for i in range(int(os.environ.get('NUM_RUNS', 53))):
      self._benchmark_train(
          '%03d_nips_mlp.pytorch' % i, self._trainer_mlp_pytorch)
      self._benchmark_simple(
          '%03d_nips_loop.pytorch' % i, self._simple_loop_pytorch)

  def benchmark_nips_ws_tfe(self):
    """Benchmarks for the NIPS workshop paper."""
    for i in range(int(os.environ.get('NUM_RUNS', 53))):
      self._benchmark_train(
          '%03d_nips_mlp.tfe_tangent' % i, self._trainer_mlp_tfe_tangent)
      self._benchmark_simple(
          '%03d_nips_loop.tfe_tangent' % i, self._simple_loop_tfe_tangent)
      self._benchmark_train(
          '%03d_nips_mlp.tfe_builtin' % i, self._trainer_mlp_tfe_builtin)
      self._benchmark_simple(
          '%03d_nips_loop.tfe_builtin' % i, self._simple_loop_tfe_builtin)

  def benchmark_nips_ws_numpy_tangent(self):
    """Benchmarks for the NIPS workshop paper."""
    for i in range(int(os.environ.get('NUM_RUNS', 53))):
      self._benchmark_train(
          '%03d_nips_mlp.numpy_tangent' % i, self._trainer_mlp_numpy_tangent)
      self._benchmark_simple(
          '%03d_nips_loop.numpy_tangent' % i, self._simple_loop_numpy_tangent)

  def benchmark_numpy_autograd(self):
    if _have_gpu():
      return  # NumPy benchmarks do not compare on GPU.
    self._benchmark_train(
        'basic_mlp.numpy_autograd', self._trainer_mlp_numpy_autograd)
    self._benchmark_train(
        'basic_rnn.numpy_autograd', self._trainer_rnn_numpy_autograd)
    self._benchmark_simple(
        'simple_loop.numpy_autograd', self._simple_loop_numpy_autograd)

  def benchmark_numpy_tangent(self):
    if _have_gpu():
      return  # NumPy benchmarks do not compare on GPU.
    self._benchmark_train(
        'basic_mlp.numpy_tangent', self._trainer_mlp_numpy_tangent)
    self._benchmark_train(
        'basic_rnn.numpy_tangent', self._trainer_rnn_numpy_tangent)
    self._benchmark_simple(
        'simple_loop.numpy_tangent', self._simple_loop_numpy_tangent)

  def benchmark_tfe_tangent(self):
    self._benchmark_train(
        'basic_mlp.tfe_tangent', self._trainer_mlp_tfe_tangent)
    self._benchmark_train(
        'basic_rnn.tfe_tangent', self._trainer_rnn_tfe_tangent)
    self._benchmark_simple(
        'simple_loop.tfe_tangent', self._simple_loop_tfe_tangent)

  def benchmark_tfe_builtin(self):
    self._benchmark_train(
        'basic_mlp.tfe_builtin', self._trainer_mlp_tfe_builtin)
    self._benchmark_train(
        'basic_rnn.tfe_builtin', self._trainer_rnn_tfe_builtin)
    self._benchmark_simple(
        'simple_loop.tfe_builtin', self._simple_loop_tfe_builtin)


class Resnet50Benchmark(_ReportingBenchmark):
  """Benchmarks for the Resnet-50 model demo."""

  def _create_inputs(self, batch_size, fmt='hwc'):
    if fmt == 'hwc':
      return self.create_random_values(batch_size, (224, 224, 3))
    elif fmt == 'cwh':
      return self.create_random_values(batch_size, (3, 224, 224))
    else:
      assert False

  def _create_labels(self, batch_size, one_hot=True):
    return self.create_random_categories(batch_size, 1000, one_hot)

  def _tag(self, name, batch_size):
    dev = 'gpu' if _have_gpu() else 'cpu'
    tag = '%s.%s.batch_%d' % (name, dev, batch_size)
    if _tag_prefix():
      tag = '%s_%s' % (_tag_prefix(), tag)
    return tag

  def _trainer_tangent(self, input_tensor, input_label):
    def loss(inputs, params, state, hparams):
      x, y = inputs
      logits = models.resnet_50(x, params, state, hparams)
      return tf.reduce_mean(softmax_crossent(logits, y))
    params, state, hparams = models.resnet_50_params(
        input_shape=input_tensor.shape.as_list(),
        classes=1000,
        bn_momentum=0.99)
    dloss = grad(loss, (1,))
    opt = train_utils.sgd()
    lr = tf.constant(0.1, dtype=tf.float32)
    init_dloss = tf.ones((), dtype=tf.float32)

    def train_one_step(params):
      dparams = dloss(
          (input_tensor, input_label), params, state, hparams, init_dloss)
      return opt(params, dparams, lr)
    return params, train_one_step

  def _trainer_pytorch(self, input_array, input_label):
    model = resnet50().cuda()
    # This will only work on GPU.
    model = torch.nn.DataParallel(model).cuda().train()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    params = model.parameters()
    optimizer = torch.optim.SGD(params, 0.1)

    def train_one_step(_):
      loss = criterion(
          model(torch.autograd.Variable(torch.Tensor(input_array))),
          torch.autograd.Variable(torch.LongTensor(input_label).cuda()))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      return None
    return model, train_one_step

  def _trainer_tangent_with_factory_optimizer(
      self, input_tensor, input_label):
    def loss(inputs, params, state, hparams):
      x, y = inputs
      logits = models.resnet_50(x, params, state, hparams)
      return tf.reduce_mean(softmax_crossent(logits, y))
    params, state, hparams = models.resnet_50_params(
        input_shape=input_tensor.shape.as_list(),
        classes=1000,
        bn_momentum=0.99)
    dloss = grad(loss, (1,))
    init_dloss = tf.ones((1,), dtype=tf.float32)

    vs = []
    train_utils.flatten_dict(params, vs)
    optimizer = tf.train.GradientDescentOptimizer(
        tf.constant(0.1, dtype=tf.float32))

    def train_one_step(_):
      dparams = dloss(
          (input_tensor, input_label), params, state, hparams, init_dloss)
      grads = []
      train_utils.flatten_dict(dparams, grads)
      optimizer.apply_gradients(zip(grads, vs))
    return None, train_one_step

  def _trainer_eager(self, input_tensor, input_label):
    def loss(inputs, params, state, hparams):
      x, y = inputs
      logits = models.resnet_50(x, params, state, hparams)
      return tf.reduce_mean(softmax_crossent(logits, y))

    params, state, hparams = models.resnet_50_params(
        input_shape=input_tensor.shape.as_list(),
        classes=1000,
        bn_momentum=0.99)

    def model_fn():
      return loss((input_tensor, input_label), params, state, hparams)
    optimizer = tf.train.GradientDescentOptimizer(
        tf.constant(0.1, dtype=tf.float32))

    def train_one_step(_):
      _, grads_and_vars = tfe.implicit_value_and_gradients(model_fn)()
      optimizer.apply_gradients(grads_and_vars)

    return None, train_one_step

  def _trainer_graph(self, input_tensor, input_label):
    params, state, hparams = models.resnet_50_params(
        input_shape=input_tensor.shape.as_list(),
        classes=1000,
        bn_momentum=0.99)
    logits = models.resnet_50(input_tensor, params, state, hparams)
    loss = tf.reduce_mean(softmax_crossent(logits, input_label))

    optimizer = tf.train.GradientDescentOptimizer(
        tf.constant(0.1, dtype=tf.float32))
    train_op = optimizer.minimize(loss)

    def train_one_step(sess):
      sess.run(train_op)

    return train_one_step

  def _benchmark_train_pytorch(self, name, trainer_factory):
    for batch_size in _train_batch_sizes():
      with tf.Graph().as_default(), tf.device(_target_device()):
        tag = self._tag(name, batch_size)
        num_burn = 5
        num_iters = _num_iterations(batch_size, train=True)
        input_tensor = self._create_inputs(batch_size, 'cwh')
        label_tensor = self._create_labels(batch_size, False)

        model, train_one_step = trainer_factory(
            input_tensor, label_tensor)

        for _ in range(num_burn):
          train_one_step(None)

        def _force_gpu_sync():
          next(model.parameters()).data.cpu()

        with _cprofile(tag):
          for _ in range(num_burn):
            train_one_step(None)
        _force_gpu_sync()
        with _sampling_profile(tag):
          for _ in range(num_burn):
            train_one_step(None)
        _force_gpu_sync()

        start = time.time()
        for _ in range(num_iters):
          train_one_step(None)
        _force_gpu_sync()
        self.report_results(tag, start, num_iters, batch_size)

  def _benchmark_train_tfe(self, name, trainer_factory):
    for batch_size in _train_batch_sizes():
      with tf.Graph().as_default(), tf.device(_target_device()):
        tag = self._tag(name, batch_size)
        num_burn = 5
        num_iters = _num_iterations(batch_size, train=True)
        input_tensor = tf.constant(self._create_inputs(batch_size))
        label_tensor = tf.constant(self._create_labels(batch_size))

        params, train_one_step = trainer_factory(
            input_tensor, label_tensor)

        def _force_gpu_sync():
          tf.constant(1.).cpu()

        for _ in range(num_burn):
          params = train_one_step(params)
          _force_gpu_sync()

        with _cprofile(tag):
          for _ in range(num_burn):
            params = train_one_step(params)
        _force_gpu_sync()
        with _sampling_profile(tag):
          for _ in range(num_burn):
            params = train_one_step(params)
        _force_gpu_sync()

        start = time.time()
        for _ in range(num_iters):
          params = train_one_step(params)
        _force_gpu_sync()
        self.report_results(tag, start, num_iters, batch_size)

  def _benchmark_train_graph(self, name, trainer_factory):
    for batch_size in _train_batch_sizes():
      with tf.Graph().as_default(), tf.device(_target_device()):
        tag = self._tag(name, batch_size)
        num_burn = 5
        num_iters = _num_iterations(batch_size, train=True)
        input_tensor = tf.constant(self._create_inputs(batch_size))
        label_tensor = tf.constant(self._create_labels(batch_size))

        train_one_step = trainer_factory(input_tensor, label_tensor)

        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          for _ in range(num_burn):
            train_one_step(sess)

          start = time.time()
          for _ in range(num_iters):
            train_one_step(sess)
          self.report_results(tag, start, num_iters, batch_size)

  def benchmark_nips_train_pytorch(self):
    for i in range(int(os.environ.get('NUM_RUNS', 11))):
      self._benchmark_train_pytorch(
          '%03d_nips_resnet50_train.pytorch' % i,
          self._trainer_pytorch)

  def benchmark_nips_train_tangent(self):
    for i in range(int(os.environ.get('NUM_RUNS', 11))):
      self._benchmark_train_tfe(
          '%03d_nips_resnet50_train.tangent' % i, self._trainer_tangent)

  def benchmark_nips_train_eager(self):
    for i in range(int(os.environ.get('NUM_RUNS', 11))):
      self._benchmark_train_tfe(
          '%03d_nips_resnet50_train.eager' % i, self._trainer_eager)

  def benchmark_nips_train_graph(self):
    for i in range(int(os.environ.get('NUM_RUNS', 11))):
      self._benchmark_train_graph(
          '%03d_nips_resnet50_train.graph' % i, self._trainer_graph)

  def benchmark_train_tangent(self):
    self._benchmark_train_tfe('resnet50_train.tangent', self._trainer_tangent)

  def benchmark_train_tangent_with_optimizer(self):
    self._benchmark_train_tfe(
        'resnet50_train.tangent_with_optimizer',
        self._trainer_tangent_with_factory_optimizer)

  def benchmark_train_eager(self):
    self._benchmark_train_tfe('resnet50_train.eager', self._trainer_eager)

  def benchmark_train_graph(self):
    self._benchmark_train_graph('resnet50_train.graph', self._trainer_graph)


if __name__ == '__main__':
  if (not os.environ.get('TF_USES_GRAPH_MODE', False)
      and not os.environ.get('USES_TORCH', False)):
    tfe.enable_eager_execution()
  tf.test.main()
