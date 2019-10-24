# Copyright (c) 2018, Xilinx, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =================================================================
"""
This script retrains Graffitist quantized networks on ImageNet (ILSVRC2012)
training set (1.2M images) using native TF and a single worker (GPU). The 
training method (ALT) is based on the paper:

"Trained Uniform Quantization for Accurate and Efficient
Neural Network Inference on Fixed-Point Hardware"
https://arxiv.org/pdf/1903.08066.pdf

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
import graffitist
import imagenet_utils as im_utils

parser = argparse.ArgumentParser(description='TensorFlow ImageNet Training Script')
parser.add_argument('--data_dir', metavar='PATH', required=True,
          help='path to dataset dir (tfrecords)')
parser.add_argument('--ckpt_dir', metavar='PATH', required=True,
          help='path to the dir containing .ckpt/.meta')
parser.add_argument('--meta_path', metavar='PATH', required=False,
          help='path to the metagraph (optional, if not in ckpt_dir)')
parser.add_argument('--image_size', type=int, default=224, metavar='N',
          help='size of input image (default: 224)')
parser.add_argument('-b_t', '--batch_size_t', type=int, default=24, metavar='N',
          help='training mini-batch size (default: 24)')
parser.add_argument('-b_v', '--batch_size_v', type=int, default=64, metavar='N',
          help='validation mini-batch size (default: 64)')


os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # This is to filter out TensorFlow INFO and WARNING logs
#os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU visible for training

# Load python libraries for custom C++/CUDA quantize kernels.
kernel_root = os.path.join(os.path.dirname(graffitist.__file__), 'kernels')
if tf.test.is_built_with_cuda() and tf.test.is_gpu_available(cuda_only=True):
  _quantize_ops_module = tf.load_op_library(os.path.join(kernel_root, 'quantize_ops_cuda.so'))
else:
  _quantize_ops_module = tf.load_op_library(os.path.join(kernel_root, 'quantize_ops.so'))

linear_quant_kernel = _quantize_ops_module.linear_quant
linear_quant_grad_kernel = _quantize_ops_module.linear_quant_gradient

# Register gradient for custom Quantize op, also loaded from the above compiled shared object.
@tf.RegisterGradient("LinearQuant")
def _linear_quant_grad(op, grad):
  grad_wrt_inputs, grad_wrt_scale = linear_quant_grad_kernel(grad, op.inputs[0], op.inputs[1],
                                         op.inputs[2], op.inputs[3], op.get_attr('rounding_mode'))
  return grad_wrt_inputs, grad_wrt_scale, None, None

# Custom gradient for folded FusedBatchNorm for quantized retraining
@tf.RegisterGradient('FoldFusedBatchNormGradient')
def _FoldFusedBatchNormGrad(op, unused_grad_y, grad_mean, grad_var, unused_1, unused_2):
  """
  Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/quantize/python/fold_batch_norms.py#L445-L453
  """
  x = op.inputs[0]
  n = tf.cast(tf.size(x) / tf.size(grad_mean), tf.float32)
  dmean_dx = grad_mean / n
  dvar_dx = 2 * grad_var * (x - op.outputs[1]) / (n - 1)
  return (dmean_dx + dvar_dx), None, None, None, None


def train(train_filenames, val_filenames):
  # Experiment #
  exp_id = 99

  # Validate before start of training
  PREVALIDATE = True

  # If learning thresholds, set True
  LEARN_TH = True

  # Disable data-augmentation for fine-tuning (enable if training from scratch)
  DATA_AUG = False

  # Threads for tf.data input pipeline map parallelization
  num_threads = 8

  # Steps per epoch
  num_train_batches_per_epoch = int(np.ceil(im_utils.num_examples_per_epoch('train') / args.batch_size_t))
  num_val_batches_per_epoch = int(np.ceil(im_utils.num_examples_per_epoch('validation') / args.batch_size_v))

  # Train for max_epochs
  max_epochs = 5
  max_steps = max_epochs * num_train_batches_per_epoch

  # Freeze BN after freeze_bn_epochs
  freeze_bn_epochs = 1
  freeze_bn_steps = freeze_bn_epochs * num_train_batches_per_epoch

  # Freeze TH incrementally starting at freeze_th_steps for every freeze_th_inc_steps
  freeze_th_steps = 1000 * (24 / args.batch_size_t)
  freeze_th_inc_steps = 50

  # Initial learning rates for weights and thresholds
  initial_lr_wt = 1e-6
  initial_lr_th = 1e-2

  # Linear scaling rule (lr ~ batch_size)
  #   REF: https://arxiv.org/pdf/1706.02677v2.pdf
  #   "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"

  # Learning rate schedule params
  lr_wt_decay_factor = 0.94
  lr_wt_decay_steps = 3000 * (24 / args.batch_size_t)
  lr_th_decay_factor = 0.5
  lr_th_decay_steps = 1000 * (24 / args.batch_size_t)

  # Intervals
  save_summary_freq = 50
  print_freq = 100
  validate_freq = 1000

  # Adam optimizer params
  adam_beta1 = 0.9
  adam_beta2 = 0.999
  adam_epsilon = 1e-08

  # Build model
  tf.compat.v1.reset_default_graph()

  # Fetch ckpt/meta paths from ckpt_dir
  ckpt = tf.train.get_checkpoint_state(os.path.dirname(args.ckpt_dir))
  if ckpt:
    ckpt_path = ckpt.model_checkpoint_path
    # Remove any global_step digits for meta path
    meta_path = re.sub('.ckpt-\d+', '.ckpt', ckpt_path) + '.meta'
  else:
    raise ValueError("No ckpt found at {}".format(args.ckpt_dir))
 
  if args.meta_path:
    meta_path = args.meta_path

  # Load model from .meta
  if tf.io.gfile.exists(ckpt_path+'.index') and tf.io.gfile.exists(meta_path):
    print("Loading model from '{}'".format(meta_path))
    saver = tf.compat.v1.train.import_meta_graph(meta_path, clear_devices=True)
  else:
    raise ValueError("No model found!")

  g = tf.compat.v1.get_default_graph()

  if re.match('.*resnet_v1_50_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("resnet_v1_50/SpatialSqueeze:0")
    output = g.get_tensor_by_name("resnet_v1_50/predictions/Softmax:0")
  elif re.match('.*resnet_v1_101_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("resnet_v1_101/SpatialSqueeze:0")
    output = g.get_tensor_by_name("resnet_v1_101/predictions/Softmax:0")
  elif re.match('.*resnet_v1_152_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("resnet_v1_152/SpatialSqueeze:0")
    output = g.get_tensor_by_name("resnet_v1_152/predictions/Softmax:0")
  elif re.match('.*inception_v1_bn_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("InceptionV1/Logits/SpatialSqueeze:0")
    output = g.get_tensor_by_name("InceptionV1/Logits/Predictions/Softmax:0")
  elif re.match('.*inception_v2_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("InceptionV2/Logits/SpatialSqueeze:0")
    output = g.get_tensor_by_name("InceptionV2/Predictions/Softmax:0")
  elif re.match('.*inception_v3_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("InceptionV3/Logits/SpatialSqueeze:0")
    output = g.get_tensor_by_name("InceptionV3/Predictions/Softmax:0")
  elif re.match('.*inception_v4_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("InceptionV4/Logits/Logits/BiasAdd_biasadd_quant/LinearQuant:0")
    output = g.get_tensor_by_name("InceptionV4/Logits/Predictions:0")
  elif re.match('.*mobilenet_v1_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("MobilenetV1/Logits/SpatialSqueeze:0")
    output = g.get_tensor_by_name("MobilenetV1/Predictions/Softmax:0")
  elif re.match('.*mobilenet_v2_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("MobilenetV2/Logits/Squeeze:0")
    output = g.get_tensor_by_name("MobilenetV2/Predictions/Softmax:0")
  elif re.match('.*vgg16_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("vgg_16/fc8/squeezed:0")
    output = logits
  elif re.match('.*vgg19_slim.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("vgg_19/fc8/squeezed:0")
    output = logits
  elif re.match('.*darknet19.*', args.ckpt_dir):
    input = g.get_tensor_by_name("darknet19/net1:0")
    logits = g.get_tensor_by_name("darknet19/softmax1/Squeeze:0")
    output = g.get_tensor_by_name("darknet19/softmax1/Softmax:0")
  elif re.match('.*resnet_v1p5_50_keras.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input_1:0")
    logits = g.get_tensor_by_name("fc1000/BiasAdd:0")
    output = g.get_tensor_by_name("activation_49/Softmax:0")
  elif re.match('.*resnet_v1p5_50_estimator.*', args.ckpt_dir):
    input = g.get_tensor_by_name("input:0")
    logits = g.get_tensor_by_name("resnet_model/dense/BiasAdd_biasadd_quant/LinearQuant:0")
    output = g.get_tensor_by_name("resnet_model/Softmax:0")
  else:
    raise ValueError("Model input/outputs unknown!")

  try:
    freeze_bn = g.get_tensor_by_name("freeze_bn:0")
  except:
    freeze_bn = tf.compat.v1.placeholder(tf.bool, shape=(), name='freeze_bn')

  # Collect batch norm update ops
  # CAUTION: Do this before adding ema, since that uses AssignMovingAvg nodes too.
  batchnorm_updates = [ g.get_tensor_by_name(node.name + ':0') for node in g.as_graph_def().node if 'AssignMovingAvg' in node.name.split('/')[-1] ]

  # Assign weights to opt_wt and thresholds to opt_th
  var_list_wt = list(filter(lambda x: 'threshold' not in x.name, tf.compat.v1.trainable_variables()))
  var_list_th = list(filter(lambda x: 'threshold' in x.name, tf.compat.v1.trainable_variables()))

  # Build dicts for incremental threshold freezing
  freeze_th_dict = {}
  if LEARN_TH and var_list_th:
    th_var_to_freeze_th_map = { var.name: [] for var in var_list_th }
    pof2_nodes = [ node for node in g.as_graph_def().node if node.name.split('/')[-1] == 'pof2' ]
    for node in pof2_nodes:
      # Get freeze_th placeholder tensor to feed into.
      freeze_th_tensor = g.get_tensor_by_name('/'.join(node.name.split('/')[:-1]+['freeze_th']) + ':0')
      # Intially feed False into freeze_th placeholders to allow training.
      freeze_th_dict[freeze_th_tensor] = False
      # pof2 nodes are outputs of threshold variables (ignoring intermediate '/read' node)
      th_var_name = '/'.join(node.input[0].split('/')[:-1]) + ':0'
      th_var_to_freeze_th_map[th_var_name].append(freeze_th_tensor)

  # Add training infrastructure
  print("Adding training ops (loss, global_step, train_op, init_op, summary_op etc)")

  # Placeholder to feed labels for computing loss and accuracy
  labels = tf.compat.v1.placeholder(tf.int64, shape=None, name='labels')

  with tf.name_scope('training'):
    # Create a variable to count the number of train() calls. This equals the
    # number of updates applied to the variables.
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Decay the learning rate exponentially based on the number of steps.
    lr_wt = tf.compat.v1.train.exponential_decay(initial_lr_wt, global_step, lr_wt_decay_steps, lr_wt_decay_factor, staircase=True)
    lr_th = tf.compat.v1.train.exponential_decay(initial_lr_th, global_step, lr_th_decay_steps, lr_th_decay_factor, staircase=True)

    # Create optimizers that performs gradient descent on weights and thresholds.
    opt_wt = tf.compat.v1.train.AdamOptimizer(lr_wt, adam_beta1, adam_beta2, adam_epsilon)
    opt_th = tf.compat.v1.train.AdamOptimizer(lr_th, adam_beta1, adam_beta2, adam_epsilon)

    # Softmax cross entropy loss with logits (uses sparse labels instead of one-hot encoded vectors)
    sce_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits)

    # In case of regularization loss, add here
    losses = []
    losses += [sce_loss]
    losses += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)

    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of sce_loss, total_loss and learned thresholds.
    ema_l = tf.train.ExponentialMovingAverage(0.99, num_updates=global_step, zero_debias=True, name='ema_l')
    ema_th = tf.train.ExponentialMovingAverage(0.9999, num_updates=global_step, zero_debias=True, name='ema_th')
    loss_averages_op = ema_l.apply(var_list=[sce_loss, total_loss])
    ema_ops = [loss_averages_op]
    if LEARN_TH and var_list_th:
      th_averages_op = ema_th.apply(var_list=var_list_th)
      ema_ops += [th_averages_op]
      ema_th_tensors = [ ema_th.average(var) for var in var_list_th ]

    # Add dependency to compute loss_averages and th_averages.
    with tf.control_dependencies(ema_ops):
      _total_loss = tf.identity(total_loss)

    # Compute gradients of total_loss wrt weights, and sce_loss wrt thresholds.
    apply_wt_gradients_op = opt_wt.minimize(_total_loss, var_list=var_list_wt, global_step=global_step)
    apply_gradient_ops = [apply_wt_gradients_op]
    th_grads_and_vars = tf.group()
    if LEARN_TH and var_list_th:
      # CAUTION: Do not provide global_step to the 2nd optimizer, else global_step would be incremented twice as fast.
      # th_grads_and_vars is a list of tuples (gradient, variable).
      th_grads_and_vars = opt_th.compute_gradients(sce_loss, var_list=var_list_th)
      apply_th_gradients_op = opt_th.apply_gradients(th_grads_and_vars)
      apply_gradient_ops += [apply_th_gradients_op]

    # Add dependency to compute and apply gradients.
    with tf.control_dependencies(apply_gradient_ops):
      train_op = tf.identity(_total_loss, name='train_op')

    # Add dependency to compute batchnorm_updates.
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    with tf.control_dependencies([batchnorm_updates_op]):
      train_op_w_bn_updates = tf.identity(train_op, name='train_op_w_bn_updates')

    # Build an initialization operation to run below.
    init_op = tf.compat.v1.global_variables_initializer()

  with tf.name_scope('accuracy'):
    # Train accuracy
    _, preds_top_5 = tf.nn.top_k(output, k=5, sorted=True)

  with tf.name_scope('data_pipeline'):
    # Build tf.data pipeline - Training & Validation
    train_features_tensor, train_labels_tensor, _ = im_utils.dataset_input_fn(train_filenames, args.ckpt_dir, args.image_size, args.batch_size_t, num_threads, shuffle=True, num_epochs=None, initializable=False, is_training=DATA_AUG)
    val_features_tensor, val_labels_tensor, val_iterator = im_utils.dataset_input_fn(val_filenames, args.ckpt_dir, args.image_size, args.batch_size_v, num_threads, shuffle=False, num_epochs=1, initializable=True, is_training=False)
    # Indices for the 1000 classes run from 0 - 999, thus
    # we subtract 1 from the labels (which run from 1 - 1000)
    # to match with the predictions. This is not needed with
    # when the background class is present (1001 classes).
    if logits.shape[1] == 1000:
      train_labels_tensor = train_labels_tensor - 1
      val_labels_tensor = val_labels_tensor - 1

  # Create a new saver with all variables to save .meta/.ckpt every best_val_acc
  new_saver = tf.compat.v1.train.Saver()

  # SUMMARIES for Tensorboard
  # Attach a scalar summary to track the learning rate.
  tf.compat.v1.summary.scalar('learning_rates/lr_wt', lr_wt)
  tf.compat.v1.summary.scalar('learning_rates/lr_th', lr_th)

  # Attach a scalar summmary to total loss and it's averaged version.
  for l in [sce_loss, total_loss]:
    loss_name = l.op.name
    # Name each loss as '(raw)' and name the moving average version of the
    # loss as the original loss name.
    tf.compat.v1.summary.scalar('losses/'+loss_name+'_raw', l)
    tf.compat.v1.summary.scalar('losses/'+loss_name+'_ema', ema_l.average(l))

  # Attach a scalar summary to track the thresholds.
  if LEARN_TH and var_list_th:
    for th in var_list_th:
      th_name = th.name
      tf.compat.v1.summary.scalar('thresholds/'+th_name+'_raw', th)
      tf.compat.v1.summary.scalar('thresholds/'+th_name+'_ema', ema_th.average(th))

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.compat.v1.summary.merge_all()

  # Output dir
  train_dir = os.path.join(os.path.dirname(args.ckpt_dir), 'train_dir_%03d'%(exp_id))
  if tf.io.gfile.exists(train_dir):
      tf.io.gfile.rmtree(train_dir)
  train_dir += '/'
  new_ckpt_path = train_dir+ckpt_path.split('/')[-1]
  new_meta_path = new_ckpt_path+'.meta'

  print("Train dir: '{}'".format(train_dir))

  # Summary writers
  train_writer = tf.compat.v1.summary.FileWriter(train_dir + '/train', g)

  # Meters to keep track of training progress
  batch_time_t = im_utils.AverageMeter()
  top1_t = im_utils.AverageMeter()
  top5_t = im_utils.AverageMeter()
  loss_t = im_utils.AverageMeter()

  # Meters to keep track of validation progress
  batch_time_v = im_utils.AverageMeter()
  top1_v = im_utils.AverageMeter()
  top5_v = im_utils.AverageMeter()

  best_val_acc = 0.0

  with tf.compat.v1.Session(graph=g) as sess:
    # Initialize new variables
    print("Initializing global variables")
    sess.run(init_op)
    print("Loading weights from '{}'".format(ckpt_path))
    saver.restore(sess, ckpt_path)

    step = sess.run(global_step)

    # VALIDATE initially, before training
    if PREVALIDATE:
      batch_time_v.reset()
      top1_v.reset()
      top5_v.reset()
      sess.run(val_iterator.initializer)
      i = 0
      while True:
        start_time = time.time()
        try:
          input_features, input_labels = sess.run([val_features_tensor, val_labels_tensor])
        except tf.errors.OutOfRangeError:
          break

        # Map predicted labels synset ordering between ILSVRC and darknet
        if re.match('.*darknet19.*', args.ckpt_dir):
          input_labels = im_utils.map_darknet_labels(input_labels, 'ilsvrc2darknet')

        preds_5 = sess.run(preds_top_5, feed_dict={input: input_features, labels: input_labels, freeze_bn: True})
        end_time = time.time()
        acc1, acc5 = im_utils.accuracy(preds_5, input_labels, topk=(1, 5))
        batch_size = input_labels.shape[0]
        top1_v.update(acc1, batch_size)
        top5_v.update(acc5, batch_size)
        batch_time_v.update(end_time-start_time, batch_size)
        print('\rVal:\t[{step:6d}/{total:6d}]\t'
           'Time {batch_time.val:7.3f} ({batch_time.avg:7.3f})\t'
           'Prec@1 {top1.val:7.3f} ({top1.avg:7.3f})\t'
           'Prec@5 {top5.val:7.3f} ({top5.avg:7.3f})'.format(
           step=i, total=num_val_batches_per_epoch, batch_time=batch_time_v,
           top1=top1_v, top5=top5_v), end='')
        i += 1
      print()
      # SAVE BEST_VAL_ACC weights to ckpt
      if top1_v.avg > best_val_acc:
        best_val_acc = top1_v.avg
        new_saver.export_meta_graph(new_meta_path, clear_devices=True, clear_extraneous_savers=True)
        new_saver.save(sess, new_ckpt_path, global_step=step, write_meta_graph=False)

    if LEARN_TH and var_list_th:
      # Initialize threshold gradient and var averages to freeze threshold with the smallest gradient.
      th_grads_vars_avg = np.zeros((len(th_grads_and_vars), len(th_grads_and_vars[0])))
      frozen_th = set()

    # This happens only once at the first data fetch
    print("Filling shuffle buffer (one-time)")
    while step < max_steps:
      # Freeze BatchNorm only after training loss seems to have converged (~1 epoch)
      freeze_bn_train = (step > freeze_bn_steps)

      # TRAIN every step
      start_time = time.time()
      input_features, input_labels = sess.run([train_features_tensor, train_labels_tensor])

      # Map predicted labels synset ordering between ILSVRC and darknet
      if re.match('.*darknet19.*', args.ckpt_dir):
        input_labels = im_utils.map_darknet_labels(input_labels, 'ilsvrc2darknet')

      loss_value, summary, step, preds_5, th_grads_vars = sess.run([train_op if freeze_bn_train else train_op_w_bn_updates,
                                                     summary_op, global_step, preds_top_5, th_grads_and_vars],
                                                     feed_dict={input: input_features, labels: input_labels, freeze_bn: freeze_bn_train, **freeze_th_dict})
      end_time = time.time()
      acc1, acc5 = im_utils.accuracy(preds_5, input_labels, topk=(1, 5))
      batch_size = input_labels.shape[0]
      loss_t.update(loss_value, batch_size)
      top1_t.update(acc1, batch_size)
      top5_t.update(acc5, batch_size)
      batch_time_t.update(end_time-start_time, batch_size)
      print('\rTrain:\t[{step:6d}/{total:6d}]\t'
         'Time {batch_time.val:7.3f} ({batch_time.avg:7.3f})\t'
         'Prec@1 {top1.val:7.3f} ({top1.avg:7.3f})\t'
         'Prec@5 {top5.val:7.3f} ({top5.avg:7.3f})\t'
         'Loss {loss.val:7.3f} ({loss.avg:7.3f})\t'
         'BestVal {best_val_acc:3.3f}'.format(
         step=step, total=max_steps, batch_time=batch_time_t,
         top1=top1_t, top5=top5_t, loss=loss_t, best_val_acc=best_val_acc), end='')
      if freeze_bn_train and batchnorm_updates:
        print('\t(BN frozen)', end='')
      if step % print_freq == 0:
        print()

      # SAVE TRAIN SUMMARIES every save_summary_freq steps
      if step % save_summary_freq == 0:
        train_writer.add_summary(summary, step)

      # FREEZE THRESHOLDS INCREMENTALLY, starting at freeze_th_steps for every freeze_th_inc_steps,
      # in order of absolute magnitude of gradients (smallest to largest) provided [ceil(curr_value) == ceil(ema_value)].
      # This basically asserts that the threshold is currently in the more popular integer bin
      # of the two bins it oscillates between.
      if LEARN_TH and var_list_th and step > freeze_th_steps and len(frozen_th) < len(var_list_th):
        th_grads_vars_avg += th_grads_vars
        if step % freeze_th_inc_steps == 0:
          th_grads_vars_avg /= freeze_th_inc_steps
          # Create a list of tuples (avg_th_grad, th_idx), sorted (lowest to highest)
          # by magnitude of gradients of threshold var.
          sorted_grads = sorted([ (np.abs(x[0]), x[1], i) for i, x in enumerate(th_grads_vars_avg) ])
          ema_thresholds = sess.run(ema_th_tensors)
          for avg_th_grad_absval, avg_th_var, th_idx in sorted_grads:
            if th_idx not in frozen_th:
              curr_th_var = th_grads_vars[th_idx][1]
              ema_th_var = ema_thresholds[th_idx]
              if np.ceil(curr_th_var) == np.ceil(ema_th_var):
                if step % print_freq != 0:
                  print()
                print("Freezing threshold ({:4d}/{:4d}): [|grad|={:.2e} | ema_val={:+.4f} | curr_val={:+.4f}] {}"
                       .format(len(frozen_th)+1, len(var_list_th), avg_th_grad_absval, ema_th_var, curr_th_var, th_var_name))
                th_var_name = var_list_th[th_idx].name
                for freeze_th_tensor in th_var_to_freeze_th_map[th_var_name]:
                  freeze_th_dict[freeze_th_tensor] = True
                # Mark this threshold as frozen.  
                frozen_th.add(th_idx)
                # Freeze one and exit until after the next freeze_th_inc_steps
                break
          # Reset average every freeze_th_inc_steps.
          th_grads_vars_avg = np.zeros((len(th_grads_and_vars), len(th_grads_and_vars[0])))

      # VALIDATE every validate_freq steps
      if step % validate_freq == 0 and step != 0:
        batch_time_v.reset()
        top1_v.reset()
        top5_v.reset()
        sess.run(val_iterator.initializer)
        i = 0
        while True:
          start_time = time.time()
          try:
            input_features, input_labels = sess.run([val_features_tensor, val_labels_tensor])
          except tf.errors.OutOfRangeError:
            break

          # Map predicted labels synset ordering between ILSVRC and darknet
          if re.match('.*darknet19.*', args.ckpt_dir):
            input_labels = im_utils.map_darknet_labels(input_labels, 'ilsvrc2darknet')

          preds_5 = sess.run(preds_top_5, feed_dict={input: input_features, labels: input_labels, freeze_bn: True})
          end_time = time.time()
          acc1, acc5 = im_utils.accuracy(preds_5, input_labels, topk=(1, 5))
          batch_size = input_labels.shape[0]
          top1_v.update(acc1, batch_size)
          top5_v.update(acc5, batch_size)
          batch_time_v.update(end_time-start_time, batch_size)
          print('\rVal:\t[{step:6d}/{total:6d}]\t'
             'Time {batch_time.val:7.3f} ({batch_time.avg:7.3f})\t'
             'Prec@1 {top1.val:7.3f} ({top1.avg:7.3f})\t'
             'Prec@5 {top5.val:7.3f} ({top5.avg:7.3f})'.format(
             step=i, total=num_val_batches_per_epoch, batch_time=batch_time_v,
             top1=top1_v, top5=top5_v), end='')
          i += 1
        print()
        # SAVE BEST_VAL_ACC weights to ckpt
        if top1_v.avg > best_val_acc:
          best_val_acc = top1_v.avg
          new_saver.export_meta_graph(new_meta_path, clear_devices=True, clear_extraneous_savers=True)
          new_saver.save(sess, new_ckpt_path, global_step=step, write_meta_graph=False)

    if best_val_acc != 0.0:
      print()
      print("Saved best model to '{}'".format(new_meta_path))
      print("Saved best weights to '{}-[step_id]'".format(new_ckpt_path))


def main():
  global args
  args = parser.parse_args()

  # ImageNet Training Dataset (TFRecords)
  num_shards = 1024
  train_filenames = []
  for shard in range(num_shards):
    file_name = 'train-%.5d-of-%.5d.tfrecord' % (shard, num_shards)
    train_filenames.append(os.path.join(args.data_dir, file_name))

  # ImageNet Validation Dataset (TFRecords)
  num_shards = 128
  val_filenames = []
  for shard in range(num_shards):
    file_name = 'validation-%.5d-of-%.5d.tfrecord' % (shard, num_shards)
    val_filenames.append(os.path.join(args.data_dir, file_name))

  # Build complete training graph (loss, optimizer, initializer, summaries etc) and train.
  train(train_filenames, val_filenames)


if __name__ == '__main__':
  main()

