# Copyright (c) 2018, Xilinx, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =================================================================
"""
This script validates pretrained (FP32) and Graffitist quantized networks on
ImageNet (ILSVRC2012) validation set (50k images).

Appropriate preprocessing is applied. Expects data to reside as TF-Records
generated from raw data using this script:
https://github.com/tensorflow/models/blob/r1.12.0/research/slim/datasets/build_imagenet_data.py

@ author: Sambhav Jain
"""

import argparse
import os
import re
import time
import numpy as np

import tensorflow as tf

# Local imports
import imagenet_utils as im_utils

parser = argparse.ArgumentParser(description='TensorFlow ImageNet Validation Script')
parser.add_argument('--data_dir', metavar='PATH', required=True,
          help='path to dataset dir (tfrecords)')
parser.add_argument('--model_dir', metavar='PATH', required=True,
          help='path to the model dir (saved_model or meta/ckpt or pb/ckpt) or path to frozen model.pb')
parser.add_argument('--image_size', type=int, default=224, metavar='N',
          help='size of input image (default: 224)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
          help='mini-batch size (default: 64)')
parser.add_argument('--gen_calib_set', dest='gen_calib_set', action='store_true',
          help='generate calibration dataset for quantization')
parser.add_argument('--calib_set_size', type=int, default=50, metavar='N',
          help='calibration dataset size (default: 50)')
parser.add_argument('--deterministic', dest='deterministic', action='store_true',
          help='sets TF_CUDNN_USE_AUTOTUNE=0 to prevent non-deterministic results (Caution: slower!)')


os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # This is to filter out TensorFlow INFO and WARNING logs
#os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU visible for validation

# Load python libraries for custom C++/CUDA quantize kernels.
kernel_root = os.path.abspath(os.path.join(__file__, '../../kernels'))
if tf.test.is_built_with_cuda() and tf.test.is_gpu_available(cuda_only=True):
  tf.load_op_library(os.path.join(kernel_root, 'quantize_ops_cuda.so'))
else:
  tf.load_op_library(os.path.join(kernel_root, 'quantize_ops.so'))


def validate(val_filenames):
  # Steps per epoch
  num_val_batches_per_epoch = int(np.ceil(im_utils.num_examples_per_epoch('validation') / args.batch_size))

  # Intervals
  print_freq = 100

  # Threads for tf.data input pipeline map parallelization
  num_threads = 8

  if args.gen_calib_set:
    # No need to load the model to generate calibration dataset
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      features, _, _ = im_utils.dataset_input_fn(val_filenames, args.model_dir, args.image_size, args.calib_set_size, num_threads, shuffle=True, is_training=False)
      input_features = sess.run(features)
      np.save(args.model_dir+'calibration_set', input_features)
      print("Saved calibration dataset to {}calibration_set.npy".format(args.model_dir))

  else:
    # Load the actual model to run validation
    saved_model_path = os.path.join(args.model_dir, 'saved_model.pb')
    saved_model_var_path = os.path.join(args.model_dir, 'variables/')
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(args.model_dir))
    if ckpt:
      ckpt_path = ckpt.model_checkpoint_path
      # Remove any global_step digits for meta path
      meta_path = re.sub('.ckpt-\d+', '.ckpt', ckpt_path) + '.meta'

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      # 1) Load from frozen model.pb
      if tf.io.gfile.exists(args.model_dir) and re.match(".*frozen.*\.pb", args.model_dir):
        print("Loading frozen model from '{}'".format(args.model_dir))
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(args.model_dir, 'rb') as f:
          graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

      # 2) Load from .ckpt and .pb
      elif ckpt and tf.io.gfile.exists(ckpt_path+'.index') and tf.io.gfile.exists(args.model_dir) and \
           re.match(".*.pb", args.model_dir):
        print("Loading model from '{}'".format(args.model_dir))
        print("Loading weights from '{}'".format(ckpt_path))
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(args.model_dir, 'rb') as f:
          graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        var_list = {}
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
        for key in reader.get_variable_to_shape_map():
          # Look for all variables in ckpt that are used by the graph
          try:
            tensor = sess.graph.get_tensor_by_name(key + ":0")
          except KeyError:
            # This tensor doesn't exist in the graph (for example it's
            # 'global_step' or a similar housekeeping element) so skip it.
            continue
          var_list[key] = tensor
        saver = tf.compat.v1.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt_path)

      # 3) Load from .ckpt and .meta
      elif ckpt and tf.io.gfile.exists(ckpt_path+'.index') and tf.io.gfile.exists(meta_path):
        print("Loading model from '{}'".format(meta_path))
        print("Loading weights from '{}'".format(ckpt_path))
        new_saver = tf.compat.v1.train.import_meta_graph(meta_path, clear_devices=True)
        new_saver.restore(sess, ckpt_path)

      # 4) Load from saved_model.pb and variables
      elif tf.saved_model.loader.maybe_saved_model_directory(args.model_dir) and \
          tf.io.gfile.exists(saved_model_path) and tf.io.gfile.exists(saved_model_var_path):
        print("Loading model from '{}'".format(saved_model_path))
        print("Loading weights from '{}'".format(saved_model_var_path))
        tf.saved_model.loader.load(sess,
                     [tf.saved_model.tag_constants.SERVING],
                     args.model_dir)

      else:
        raise ValueError("No model found!")

      g = tf.compat.v1.get_default_graph()

      if re.match('.*resnet_v1_50_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("resnet_v1_50/predictions/Softmax:0")
      elif re.match('.*resnet_v1_101_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("resnet_v1_101/predictions/Softmax:0")
      elif re.match('.*resnet_v1_152_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("resnet_v1_152/predictions/Softmax:0")
      elif re.match('.*inception_v1_bn_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("InceptionV1/Logits/Predictions/Softmax:0")
      elif re.match('.*inception_v2_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("InceptionV2/Predictions/Softmax:0")
      elif re.match('.*inception_v3_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("InceptionV3/Predictions/Softmax:0")
      elif re.match('.*inception_v4_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("InceptionV4/Logits/Predictions:0")
      elif re.match('.*mobilenet_v1_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("MobilenetV1/Predictions/Softmax:0")
      elif re.match('.*mobilenet_v2_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("MobilenetV2/Predictions/Softmax:0")
      elif re.match('.*vgg16_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("vgg_16/fc8/squeezed:0")
      elif re.match('.*vgg19_slim.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("vgg_19/fc8/squeezed:0")
      elif re.match('.*darknet19.*', args.model_dir):
        input = g.get_tensor_by_name("darknet19/net1:0")
        output = g.get_tensor_by_name("darknet19/softmax1/Softmax:0")
      elif re.match('.*inception_v1_bn_keras.*', args.model_dir):
        input = g.get_tensor_by_name("input_1:0")
        output = g.get_tensor_by_name("Predictions/Softmax:0")
      elif re.match('.*resnet_v1p5_50_keras.*', args.model_dir):
        input = g.get_tensor_by_name("input_1:0")
        output = g.get_tensor_by_name("activation_49/Softmax:0")
      elif re.match('.*caffe2tf.*', args.model_dir):
        input = g.get_tensor_by_name("input:0")
        output = g.get_tensor_by_name("prob:0")
      else:
        raise ValueError("Model input/outputs unknown!")
      
      # Meters to keep track of validation progress
      batch_time = im_utils.AverageMeter()
      top1 = im_utils.AverageMeter()
      top5 = im_utils.AverageMeter()

      _, preds_top_5 = tf.nn.top_k(output, k=5, sorted=True)
      features, labels, _ = im_utils.dataset_input_fn(val_filenames, args.model_dir, args.image_size, args.batch_size, num_threads, shuffle=False, num_epochs=1, is_training=False)

      i = 0
      end = time.time()
      while True:
        try:
          input_features, input_labels = sess.run([features, labels])
        except tf.errors.OutOfRangeError:
          break

        all_preds, preds_5 = sess.run([output, preds_top_5], {input: input_features})

        # Indices for the 1000 classes run from 0 - 999, thus
        # we subtract 1 from the labels (which run from 1 - 1000)
        # to match with the predictions. This is not needed with
        # when the background class is present (1001 classes)
        if all_preds.shape[1] == 1000:
          input_labels -= 1

        # Map predicted labels synset ordering between ILSVRC and darknet
        if re.match('.*darknet19.*', args.model_dir):
          input_labels = im_utils.map_darknet_labels(input_labels, 'ilsvrc2darknet')

        acc1, acc5 = im_utils.accuracy(preds_5, input_labels, topk=(1, 5))
        batch_size = input_labels.shape[0]

        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        print('\rVal:\t[{step:6d}/{total:6d}]\t'
           'Time {batch_time.val:7.3f} ({batch_time.avg:7.3f})\t'
           'Prec@1 {top1.val:7.3f} ({top1.avg:7.3f})\t'
           'Prec@5 {top5.val:7.3f} ({top5.avg:7.3f})'.format(
           step=i, total=num_val_batches_per_epoch, batch_time=batch_time,
           top1=top1, top5=top5), end='')
        if i % print_freq == 0:
          print('')

        i += 1

      result = "model_dir={} prec@1={:.3f} prec@5={:.3f}".format(
            args.model_dir, top1.avg, top5.avg)
      print('\n', result)
      with open('pretrained_results.txt', 'a') as f:
        f.write(result + '\n')


def main():
  global args
  args = parser.parse_args()

  # References
  # Non-determinism is expected with GPU/CUDA/cuDNN and with Eigen/TF atomics
  #   https://github.com/tensorflow/tensorflow/issues/2732#issuecomment-224661591
  #   https://github.com/tensorflow/tensorflow/issues/2732#issuecomment-224947963
  #   https://github.com/tensorflow/tensorflow/issues/22398#issuecomment-423704157
  #   https://github.com/tensorflow/tensorflow/issues/22398#issuecomment-431211132
  #   https://github.com/tensorflow/tensorflow/issues/2732#issuecomment-224633688
  # About TF_CUDNN_USE_AUTOTUNE
  #   https://github.com/tensorflow/tensorflow/issues/5048#issuecomment-254672224
  #   https://github.com/tensorflow/tensorflow/issues/2732#issuecomment-366824564
  # Non-deterministic clip_by_value / reduce_mean / reduce_sum
  #   https://github.com/tensorflow/tensorflow/issues/12871#issuecomment-401593160
  #   https://www.twosigma.com/insights/article/a-workaround-for-non-determinism-in-tensorflow/
  #   http://jkschin.com/2017/06/30/non-determinism.html
  if args.deterministic:
    # This is to prevent non-deterministic results due to atomic ops with Eigen for CUDA.
    os.environ["TF_CUDNN_USE_AUTOTUNE"]="0"

  # ImageNet Validation Dataset (TFRecords)
  num_shards = 128
  val_filenames = []
  for shard in range(num_shards):
    file_name = 'validation-%.5d-of-%.5d.tfrecord' % (shard, num_shards)
    val_filenames.append(os.path.join(args.data_dir, file_name))

  # Run validation
  validate(val_filenames)


if __name__ == '__main__':
  main()

