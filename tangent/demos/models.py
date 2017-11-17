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
"""Model builders."""
from __future__ import absolute_import

import math

from tangent.demos.common_kernels import avgpool2d
from tangent.demos.common_kernels import batch_norm
from tangent.demos.common_kernels import conv2d
from tangent.demos.common_kernels import dense
from tangent.demos.common_kernels import maxpool2d
from tangent.demos.common_kernels import relu

import tensorflow as tf


def stack_params(*params_list):
  params_ = {}
  initial_state_ = {}
  hparams_ = {}
  for i in range(len(params_list)):
    params_[i], initial_state_[i], hparams_[i] = params_list[i]

  # Convention: the final element is considered to be the "output layer" of
  # the block.
  hparams_['output_shape'] = params_list[-1][2]['output_shape']

  return params_, initial_state_, hparams_


_autonumber = [0]


def autonumber(tag):
  _autonumber[0] += 1
  return '%s_%d' % (tag, _autonumber[0])


def dense_layer_params(
    input_shape,
    num_units,
    activation):
  params_ = {}
  params_['weight'] = tf.get_variable(
      autonumber('weight'),
      shape=(input_shape[-1], num_units),
      initializer=tf.contrib.layers.xavier_initializer())
  params_['bias'] = tf.get_variable(
      autonumber('bias'),
      shape=(1, num_units),
      initializer=tf.zeros_initializer())

  initial_state_ = {}

  hparams_ = {}
  hparams_['input_shape'] = input_shape
  hparams_['output_shape'] = input_shape[:-1] + (num_units,)
  hparams_['activation'] = activation

  return params_, initial_state_, hparams_


def dense_layer(x, params, state, hparams):
  x = dense(x, params['weight'], params['bias'])
  if hparams['activation'] == 'relu':
    x = relu(x)
  return x


def pooling_layer_params(
    input_shape,
    pooling,
    size,
    strides):
  if pooling in ('max', 'avg'):
    output_shape = (
        input_shape[0],
        int(math.ceil(float(input_shape[1]) / strides)),
        int(math.ceil(float(input_shape[2]) / strides)),
        input_shape[3],
    )
  else:
    output_shape = input_shape

  params_ = {}

  initial_state_ = {}

  hparams_ = {}
  hparams_['pooling'] = pooling
  hparams_['size'] = (size, size)
  hparams_['strides'] = (1, strides, strides, 1)
  hparams_['output_shape'] = output_shape

  return params_, initial_state_, hparams_


def pooling_layer(x, params, state, hparams):
  if hparams['pooling'] == 'avg':
    x = avgpool2d(
        x,
        hparams['size'],
        strides=hparams['strides'],
        padding='SAME')
  if hparams['pooling'] == 'max':
    x = maxpool2d(
        x,
        hparams['size'],
        strides=hparams['strides'],
        padding='SAME')
  return x


def resnet_conv_layer_params(
    input_shape,
    num_filters,
    kernel_size,
    strides,
    bn_momentum,
    activation):
  bn_activation_shape = (
      1,
      int(math.ceil(float(input_shape[1]) / strides)),
      int(math.ceil(float(input_shape[2]) / strides)),
      num_filters,
  )
  output_shape = (input_shape[0],) + bn_activation_shape[1:]

  params_ = {}
  params_['kernel'] = tf.get_variable(
      autonumber('kernel'),
      shape=(kernel_size, kernel_size, input_shape[-1], num_filters),
      initializer=tf.contrib.layers.xavier_initializer())
  params_['bias'] = tf.get_variable(
      autonumber('bias'),
      shape=(1, 1, 1, num_filters),
      initializer=tf.zeros_initializer())
  params_['bn'] = {}
  params_['bn']['offset'] = tf.get_variable(
      autonumber('offset'),
      shape=(num_filters,),
      initializer=tf.zeros_initializer())
  params_['bn']['scale'] = tf.get_variable(
      autonumber('scale'),
      shape=(num_filters,),
      initializer=tf.ones_initializer())

  initial_state_ = {'bn': {}}
  initial_state_['bn']['mean'] = tf.get_variable(
      autonumber('bn_mean'),
      shape=bn_activation_shape,
      initializer=tf.zeros_initializer(),
      trainable=False)
  initial_state_['bn']['variance'] = tf.get_variable(
      autonumber('bn_variance'),
      shape=bn_activation_shape,
      initializer=tf.ones_initializer(),
      trainable=False)

  hparams_ = {}
  hparams_['strides'] = strides
  hparams_['bn_momentum'] = tf.constant(bn_momentum)
  hparams_['activation'] = activation
  hparams_['update_bn'] = True
  hparams_['output_shape'] = output_shape

  return params_, initial_state_, hparams_


def resnet_conv_layer(x, params, state, hparams):
  # TODO: Remove the default params.
  x = conv2d(
      x,
      params['kernel'],
      strides=(1, hparams['strides'], hparams['strides'], 1),
      padding='SAME')
  x = tf.add(x, params['bias'])
  x, moving_mean, moving_var = batch_norm(
      x,
      state['bn']['mean'],
      state['bn']['variance'],
      hparams['bn_momentum'],
      params['bn']['offset'],
      params['bn']['scale'],
      (0, 1, 2),
      hparams['update_bn'])
  # TODO: It would be cleaner to return the new state.
  # TODO: update_bn should sit in state or alternatively resnet_conv_layer(x, params, state, hparams, training=True)
  if hparams['update_bn']:
    state['bn']['mean'] = moving_mean
    state['bn']['variance'] = moving_var
  if hparams['activation'] == 'relu':
    x = relu(x)
  # TODO: return x, (moving_mean, moving_var) should have worked.
  return x


def resnet_conv_block_params(
    input_shape, num_filters, kernel_size, strides, bn_momentum, activation):
  c0 = resnet_conv_layer_params(
      input_shape=input_shape,
      num_filters=num_filters[0],
      kernel_size=1,
      strides=strides,
      bn_momentum=bn_momentum,
      activation=activation)
  c1 = resnet_conv_layer_params(
      input_shape=c0[2]['output_shape'],
      num_filters=num_filters[1],
      kernel_size=kernel_size,
      strides=1,
      bn_momentum=bn_momentum,
      activation=activation)
  c2 = resnet_conv_layer_params(
      input_shape=c1[2]['output_shape'],
      num_filters=num_filters[2],
      kernel_size=1,
      strides=1,
      bn_momentum=bn_momentum,
      activation='none')
  cs = resnet_conv_layer_params(
      input_shape=input_shape,
      num_filters=num_filters[2],
      kernel_size=1,
      strides=strides,
      bn_momentum=bn_momentum,
      activation='none')
  return stack_params(c0, c1, c2, cs)


def resnet_conv_block(x, params, state, hparams):
  xd = resnet_conv_layer(x, params[0], state[0], hparams[0])
  xd = resnet_conv_layer(xd, params[1], state[1], hparams[1])
  xd = resnet_conv_layer(xd, params[2], state[2], hparams[2])
  xs = resnet_conv_layer(x, params[3], state[3], hparams[3])
  x = tf.add(xd, xs)

  x = relu(x)
  return x


def resnet_identity_block_params(
    input_shape, num_filters, kernel_size, bn_momentum, activation):
  c0 = resnet_conv_layer_params(
      input_shape=input_shape,
      num_filters=num_filters[0],
      kernel_size=1,
      strides=1,
      bn_momentum=bn_momentum,
      activation=activation)
  c1 = resnet_conv_layer_params(
      input_shape=c0[2]['output_shape'],
      num_filters=num_filters[1],
      kernel_size=kernel_size,
      strides=1,
      bn_momentum=bn_momentum,
      activation=activation)
  c2 = resnet_conv_layer_params(
      input_shape=c1[2]['output_shape'],
      num_filters=num_filters[2],
      kernel_size=1,
      strides=1,
      bn_momentum=bn_momentum,
      activation='none')
  return stack_params(c0, c1, c2)


def resnet_identity_block(x, params, state, hparams):
  xd = resnet_conv_layer(x, params[0], state[0], hparams[0])
  xd = resnet_conv_layer(xd, params[1], state[1], hparams[1])
  xd = resnet_conv_layer(xd, params[2], state[2], hparams[2])
  x = tf.add(x, xd)
  return x


def resnet_50_params(input_shape, classes, bn_momentum):
  c1 = resnet_conv_layer_params(
      input_shape=input_shape,
      num_filters=64,
      kernel_size=7,
      strides=2,
      bn_momentum=bn_momentum,
      activation='relu')

  p1 = pooling_layer_params(
      input_shape=c1[2]['output_shape'],
      pooling='max',
      size=3,
      strides=2)

  l2a = resnet_conv_block_params(
      input_shape=p1[2]['output_shape'],
      num_filters=[64, 64, 256],
      kernel_size=3,
      strides=1,
      bn_momentum=bn_momentum,
      activation='relu')
  l2b = resnet_identity_block_params(
      input_shape=l2a[2]['output_shape'],
      num_filters=[64, 64, 256],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')
  l2c = resnet_identity_block_params(
      input_shape=l2b[2]['output_shape'],
      num_filters=[64, 64, 256],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')

  l3a = resnet_conv_block_params(
      input_shape=l2c[2]['output_shape'],
      num_filters=[128, 128, 512],
      kernel_size=3,
      strides=2,
      bn_momentum=bn_momentum,
      activation='relu')
  l3b = resnet_identity_block_params(
      input_shape=l3a[2]['output_shape'],
      num_filters=[128, 128, 512],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')
  l3c = resnet_identity_block_params(
      input_shape=l3b[2]['output_shape'],
      num_filters=[128, 128, 512],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')
  l3d = resnet_identity_block_params(
      input_shape=l3c[2]['output_shape'],
      num_filters=[128, 128, 512],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')

  l4a = resnet_conv_block_params(
      input_shape=l3d[2]['output_shape'],
      num_filters=[256, 256, 1024],
      kernel_size=3,
      strides=2,
      bn_momentum=bn_momentum,
      activation='relu')
  l4b = resnet_identity_block_params(
      input_shape=l4a[2]['output_shape'],
      num_filters=[256, 256, 1024],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')
  l4c = resnet_identity_block_params(
      input_shape=l4b[2]['output_shape'],
      num_filters=[256, 256, 1024],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')
  l4d = resnet_identity_block_params(
      input_shape=l4c[2]['output_shape'],
      num_filters=[256, 256, 1024],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')
  l4e = resnet_identity_block_params(
      input_shape=l4d[2]['output_shape'],
      num_filters=[256, 256, 1024],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')
  l4f = resnet_identity_block_params(
      input_shape=l4e[2]['output_shape'],
      num_filters=[256, 256, 1024],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')

  l5a = resnet_conv_block_params(
      input_shape=l4f[2]['output_shape'],
      num_filters=[512, 512, 2048],
      kernel_size=3,
      strides=2,
      bn_momentum=bn_momentum,
      activation='relu')
  l5b = resnet_identity_block_params(
      input_shape=l5a[2]['output_shape'],
      num_filters=[512, 512, 2048],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')
  l5c = resnet_identity_block_params(
      input_shape=l5b[2]['output_shape'],
      num_filters=[512, 512, 2048],
      kernel_size=3,
      bn_momentum=bn_momentum,
      activation='relu')

  p2 = pooling_layer_params(
      input_shape=l5c[2]['output_shape'],
      pooling='avg',
      size=7,
      strides=7)

  conv_shape = p2[2]['output_shape']
  dense_input_shape = (
      -1,
      conv_shape[1] * conv_shape[2] * conv_shape[3])
  d = dense_layer_params(
      input_shape=dense_input_shape,
      num_units=classes,
      activation='none')


  return stack_params(
      c1,
      p1,
      l2a, l2b, l2c,
      l3a, l3b, l3c, l3d,
      l4a, l4b, l4c, l4d, l4e, l4f,
      l5a, l5b, l5c,
      p2,
      d,
  )


def resnet_50(x, params, state, hparams):
  # c1
  x = resnet_conv_layer(x, params[0], state[0], hparams[0])

  # p1
  x = pooling_layer(x, params[1], state[1], hparams[1])

  # l2
  x = resnet_conv_block(x, params[2], state[2], hparams[2])
  x = resnet_identity_block(x, params[3], state[3], hparams[3])
  x = resnet_identity_block(x, params[4], state[4], hparams[4])

  # l3
  x = resnet_conv_block(x, params[5], state[5], hparams[5])
  x = resnet_identity_block(x, params[6], state[6], hparams[6])
  x = resnet_identity_block(x, params[7], state[7], hparams[7])
  x = resnet_identity_block(x, params[8], state[8], hparams[8])

  # l4
  x = resnet_conv_block(x, params[9], state[9], hparams[9])
  x = resnet_identity_block(x, params[10], state[10], hparams[10])
  x = resnet_identity_block(x, params[11], state[11], hparams[11])
  x = resnet_identity_block(x, params[12], state[12], hparams[12])
  x = resnet_identity_block(x, params[13], state[13], hparams[13])
  x = resnet_identity_block(x, params[14], state[14], hparams[14])

  # l5
  x = resnet_conv_block(x, params[15], state[15], hparams[15])
  x = resnet_identity_block(x, params[16], state[16], hparams[16])
  x = resnet_identity_block(x, params[17], state[17], hparams[17])

  # p
  x = pooling_layer(x, params[18], state[18], hparams[18])

  # d
  x = tf.reshape(x, (-1, hparams[19]['input_shape'][1]))
  x = dense_layer(x, params[19], state[19], hparams[19])

  return x
