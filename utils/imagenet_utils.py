# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2018, Xilinx, Inc. All rights reserved.
#
# Changes
# - self-contained data pre-processing utils for ImageNet networks
# - generic dataset_input_fn configurable for training or inference
# - mapping function for darknet vs ILSVRC synset ordering
# - accuracy meters
#
# =================================================================
"""
Utilities for training/validation on ImageNet.

Reference:
https://github.com/tensorflow/models/tree/r1.12.0/research/slim/preprocessing

@author: Sambhav Jain
"""

import os
import re
import numpy as np

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

def num_examples_per_epoch(mode):
  """Returns the number of examples in the data set."""
  if mode == 'train':
    return 1281167
  if mode == 'validation':
    return 50000

# DATA PRE-PROCESSING
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

# Aspect-ratio preserving resize
_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def map_darknet_labels(labels, order):
  """
  Map predicted labels synset ordering between ILSVRC and darknet.
  """
  metadata_root = os.path.abspath(os.path.join(__file__, '../metadata'))
  ilsvrc_synset_path = os.path.join(metadata_root, 'imagenet_lsvrc_2015_synsets.txt')
  darknet_synset_path = os.path.join(metadata_root, 'darknet_lsvrc_synsets.txt')

  ilsvrc_f = open(ilsvrc_synset_path, 'r')
  darknet_f = open(darknet_synset_path, 'r')
  ilsvrc_synsets = [ x.strip('\n') for x in ilsvrc_f.readlines() ]
  darknet_synsets = [ x.strip('\n') for x in darknet_f.readlines() ]
  ilsvrc_f.close()
  darknet_f.close()

  if order == 'darknet2ilsvrc':
    map_from_darknet_to_ilsvrc = [ ilsvrc_synsets.index(x) for x in darknet_synsets ]
    map_f = np.vectorize(lambda x: map_from_darknet_to_ilsvrc[x])
  elif order == 'ilsvrc2darknet':
    map_from_ilsvrc_to_darknet = [ darknet_synsets.index(x) for x in ilsvrc_synsets ]
    map_f = np.vectorize(lambda x: map_from_ilsvrc_to_darknet[x])
  else:
    raise ValueError("Pick map_darknet_labels order from: ['darknet2ilsvrc', 'ilsvrc2darknet']")

  return map_f(labels)


# Reference:
# https://github.com/tensorflow/models/blob/r1.12.0/research/slim/preprocessing/vgg_preprocessing.py
# -----------------------------------------------------------------------------------------------------
def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.
  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.
  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.
  Returns:
    the cropped (and resized) image.
  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), tf.int32)

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)

def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.
  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:
    image, depths, normals = _random_crop([image, depths, normals], 120, 150)
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.
  Returns:
    the image_list with cropped images.
  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]

def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.
  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
  Note that the rank of `image` must be known.
  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
  Returns:
    the centered image.
  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.
  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.cast(tf.math.rint(height * scale), tf.int32)
  new_width = tf.cast(tf.math.rint(width * scale), tf.int32)
  return new_height, new_width

def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.
  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.compat.v1.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image
# -----------------------------------------------------------------------------------------------------


# Reference:
# https://github.com/tensorflow/models/blob/r1.12.0/research/slim/preprocessing/inception_preprocessing.py
# -----------------------------------------------------------------------------------------------------
def _apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

def _distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def _distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.
  See `tf.image.sample_distorted_bounding_box` for more documentation.
  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox
# -----------------------------------------------------------------------------------------------------


# Reference:
# https://github.com/tensorflow/models/blob/r1.12.0/research/slim/preprocessing/inception_preprocessing.py
def inception_preprocess_input_fn(image, output_height, output_width, is_training=False, fast_mode=True):
  if is_training:
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)   # Returns fp32 (with range [0, 1))
    distorted_image, distorted_bbox = _distorted_bounding_box_crop(image, bbox)
    # Restore the shape since the dynamic slice based upon the bbox_size loses the third dimension.
    distorted_image.set_shape([None, None, 3])
    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.
    # We select only 1 case for fast_mode bilinear.
    num_resize_cases = 1 if fast_mode else 4
    distorted_image = _apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=num_resize_cases)
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # Randomly distort the colors. There are 1 or 4 ways to do it.
    num_distort_cases = 1 if fast_mode else 4
    distorted_image = _apply_with_random_selector(
        distorted_image,
        lambda x, ordering: _distort_color(x, ordering, fast_mode),
        num_cases=num_distort_cases)
    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)
    image = distorted_image
  else:
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)   # Returns fp32 (with range [0, 1))
    image = tf.image.central_crop(image, central_fraction=0.875)   # Crop central region containing 87.5% of original image
    image = tf.expand_dims(image, 0)
    image = tf.compat.v1.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
  return image

# Reference:
# https://github.com/tensorflow/models/blob/r1.12.0/research/slim/preprocessing/vgg_preprocessing.py
def vgg_preprocess_input_fn(image, output_height, output_width, is_training=False):
  if is_training:
    resize_side = tf.random_uniform([], minval=_RESIZE_SIDE_MIN, maxval=_RESIZE_SIDE_MAX+1, dtype=tf.int32)
    image = _aspect_preserving_resize(image, resize_side)
    image = _random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
  else:
    resize_side = _RESIZE_SIDE_MIN
    image = _aspect_preserving_resize(image, resize_side)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
  return image

def caffe2tf_preprocess_input_fn(image, output_height, output_width, is_training=False):
  if is_training:
    resize_side = tf.random_uniform([], minval=_RESIZE_SIDE_MIN, maxval=_RESIZE_SIDE_MAX+1, dtype=tf.int32)
    image = _aspect_preserving_resize(image, resize_side)
    image = _random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.reverse(image, axis=[-1])   # RGB -> BGR ;  image dims: (H, W, C)
    image = _mean_image_subtraction(image, [_B_MEAN, _G_MEAN, _R_MEAN])
  else:
    resize_side = _RESIZE_SIDE_MIN
    image = _aspect_preserving_resize(image, resize_side)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.reverse(image, axis=[-1])   # RGB -> BGR ;  image dims: (H, W, C)
    image = _mean_image_subtraction(image, [_B_MEAN, _G_MEAN, _R_MEAN])
  return image

def inception_caffe2tf_preprocess_input_fn(image, output_height, output_width, is_training=False):
  if is_training:
    raise ValueError('Training mode data preprocessing not implemented!')
  else:
    image = tf.to_float(image)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.image.resize_images(image, [output_height, output_width])  # Default = bilinear
    image = tf.reverse(image, axis=[-1])   # RGB -> BGR ;  image dims: (H, W, C)
    image = _mean_image_subtraction(image, [_B_MEAN, _G_MEAN, _R_MEAN])
  return image

def mobilenet_caffe2tf_preprocess_input_fn(image, output_height, output_width, is_training=False):
  if is_training:
    raise ValueError('Training mode data preprocessing not implemented!')
  else:
    image = tf.to_float(image)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.image.resize_images(image, [output_height, output_width])  # Default = bilinear
    image = tf.reverse(image, axis=[-1])   # RGB -> BGR ;  image dims: (H, W, C)
    image = _mean_image_subtraction(image, [_B_MEAN, _G_MEAN, _R_MEAN])
    image = tf.multiply(image, 0.017, name='norm_const')
  return image

def darknet_preprocess_input_fn(image, output_height, output_width, is_training=False):
  if is_training:
    raise ValueError('Training mode data preprocessing not implemented!')
  else:
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)   # Returns fp32 (with range [0, 1))
    image = tf.image.central_crop(image, central_fraction=0.875)   # Crop central region containing 87.5% of original image
    image = tf.expand_dims(image, 0)
    image = tf.compat.v1.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
    image = tf.squeeze(image, [0])
  return image


def dataset_input_fn(filenames, model_dir, image_size, batch_size, num_threads, shuffle=False, num_epochs=None, initializable=False, is_training=False):
  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def _parse_function(example_proto):
    features = {
      'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=""),
      'image/class/label': tf.io.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
  
    image = tf.image.decode_jpeg(parsed_features['image/encoded'], channels=3)  # Returns uint8 (with range [0, 255])
    if re.match('.*inception.*slim.*', model_dir):
      # Center crop, resize, scale and shift [0, 255] -> [0, 1] -> [-1, 1]
      image = inception_preprocess_input_fn(image, image_size, image_size, is_training)
    elif re.match('.*mobilenet.*slim.*', model_dir):
      # Center crop, resize, scale and shift [0, 255] -> [0, 1] -> [-1, 1]
      image = inception_preprocess_input_fn(image, image_size, image_size, is_training)
    elif re.match('.*vgg.*slim.*', model_dir):
      # Center crop, aspect preserving resize, mean subtraction
      image = vgg_preprocess_input_fn(image, image_size, image_size, is_training)
    elif re.match('.*resnet_v1.*slim.*', model_dir):
      # Center crop, aspect preserving resize, mean subtraction
      image = vgg_preprocess_input_fn(image, image_size, image_size, is_training)
    elif re.match('.*resnet_v1p5.*keras.*', model_dir):
      # Center crop, aspect preserving resize, mean subtraction
      image = vgg_preprocess_input_fn(image, image_size, image_size, is_training)
    elif re.match('.*darknet19.*', model_dir):
      # Center crop, resize, normalize [0, 255] -> [0, 1]
      image = darknet_preprocess_input_fn(image, image_size, image_size, is_training)
    elif re.match('.*inception.*keras.*', model_dir):
      # Center crop, resize, scale and shift [0, 255] -> [0, 1] -> [-1, 1]
      image = inception_preprocess_input_fn(image, image_size, image_size, is_training)
    elif re.match('.*inception.*caffe2tf.*', model_dir):
      # Center crop, resize, RGB->BGR, mean subtraction
      image = inception_caffe2tf_preprocess_input_fn(image, image_size, image_size, is_training)
    elif re.match('.*mobilenet.*caffe2tf.*', model_dir):
      # Center crop, resize, RGB->BGR, mean subtraction, scale by norm_const=0.017
      image = mobilenet_caffe2tf_preprocess_input_fn(image, image_size, image_size, is_training)
    elif re.match('.*caffe2tf.*', model_dir):
      # Center crop, aspect preserving resize, RGB->BGR, mean subtraction
      image = caffe2tf_preprocess_input_fn(image, image_size, image_size, is_training)
    else:
      raise ValueError("Data pre-processing unknown!")

    label = tf.cast(parsed_features['image/class/label'], tf.int32)
    return image, label

  with tf.device('/cpu:0'):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=8)
    # Ignore corrupt TFRecords (e.g. DataLossError)
    #   Reference:
    #     https://github.com/tensorflow/tensorflow/issues/13463
    #
    # Caution: Use ignore_errors only with patched TF 1.12
    # to avoid infinite loop / indefinite stalling with corrupt TFRecord.
    #   References:
    #     https://github.com/tensorflow/tensorflow/issues/25700
    #     https://github.com/tensorflow/tensorflow/pull/25705
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=1)
    if initializable:
      iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    else:
      iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    features, labels = iterator.get_next()
  return features, labels, iterator


def accuracy(predictions, labels, topk=(1,)):
  max_k = max(topk)
  batch_size = labels.shape[0]                # This is N

  predictons = np.array(predictions)          # (N, 5) for top-5
  labels = np.array(labels).reshape(-1, 1)    # (N, 1)
  is_correct = np.equal(predictions, labels)  # (N, 5)

  result = []
  for k in topk:
    result.append(np.sum(is_correct[:, :k]) * 100.0 / batch_size)
  return result


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

