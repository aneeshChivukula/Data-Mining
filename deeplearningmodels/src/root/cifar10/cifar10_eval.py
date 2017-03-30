# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import os

# from tensorflow.models.image.cifar10 import cifar10
from root.cifar10 import cifar10

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
#                            """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_dir', '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
# tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
#                            """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
# tf.app.flags.DEFINE_integer('num_examples', 10000,
#                             """Number of examples to run.""")
# tf.app.flags.DEFINE_integer('num_examples', 1500,
#                             """Number of examples to run.""")
tf.app.flags.DEFINE_integer('num_examples', 810,
                            """Number of examples to evaluate.""")
# tf.app.flags.DEFINE_boolean('run_once', False,
#                          """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('max_iter_eval', 100,
                         """Set max_iter to get sufficient mix of positive and negative classes in testing CNN and training GA.""")
tf.app.flags.DEFINE_boolean('numdecimalplaces', 4,
                         """Number of decimal places to retain in the performance metrics.""")


def eval_once(saver, summary_writer, top_k_op, summary_op,variables_to_restore,logits,labels):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return


#     print(os.path.join(FLAGS.data_dir, 'conv1-weights.npy'))
#     print(sess.run(variables_to_restore['conv1/weights/ExponentialMovingAverage']))

#     np.save(os.path.join(FLAGS.out_dir, 'conv1-weights.npy'), sess.run(variables_to_restore['conv1/weights/ExponentialMovingAverage']))
#     np.save(os.path.join(FLAGS.out_dir, 'conv1-biases.npy'), sess.run(variables_to_restore['conv1/biases/ExponentialMovingAverage']))
#     np.save(os.path.join(FLAGS.out_dir, 'conv2-weights.npy'), sess.run(variables_to_restore['conv2/weights/ExponentialMovingAverage']))
#     np.save(os.path.join(FLAGS.out_dir, 'conv2-biases.npy'), sess.run(variables_to_restore['conv2/biases/ExponentialMovingAverage']))
#     np.save(os.path.join(FLAGS.out_dir, 'local3-weights.npy'), sess.run(variables_to_restore['local3/weights/ExponentialMovingAverage']))
#     np.save(os.path.join(FLAGS.out_dir, 'local3-biases.npy'), sess.run(variables_to_restore['local3/biases/ExponentialMovingAverage']))
#     np.save(os.path.join(FLAGS.out_dir, 'local4-weights.npy'), sess.run(variables_to_restore['local4/weights/ExponentialMovingAverage']))
#     np.save(os.path.join(FLAGS.out_dir, 'local4-biases.npy'), sess.run(variables_to_restore['local4/biases/ExponentialMovingAverage']))
#     np.save(os.path.join(FLAGS.out_dir, 'softmax_linear-weights.npy'), sess.run(variables_to_restore['softmax_linear/weights/ExponentialMovingAverage']))
#     np.save(os.path.join(FLAGS.out_dir, 'softmax_linear-biases.npy'), sess.run(variables_to_restore['softmax_linear/biases/ExponentialMovingAverage']))
    
#     print('Separating here')
#     print(np.load(os.path.join(FLAGS.data_dir, 'conv1-weights.npy')).shape)
#     print('sess.run(variables_to_restore) conv1/weights/ExponentialMovingAverage',sess.run(variables_to_restore['conv1/weights/ExponentialMovingAverage']))
#     print('sess.run(variables_to_restore) conv1/biases/ExponentialMovingAverage',sess.run(variables_to_restore['conv1/biases/ExponentialMovingAverage']))
#     print('sess.run(variables_to_restore) conv2/weights/ExponentialMovingAverage',sess.run(variables_to_restore['conv2/weights/ExponentialMovingAverage']))
#     print('sess.run(variables_to_restore) conv2/biases/ExponentialMovingAverage',sess.run(variables_to_restore['conv2/biases/ExponentialMovingAverage']))
#     print('sess.run(variables_to_restore) local3/weights/ExponentialMovingAverage',sess.run(variables_to_restore['local3/weights/ExponentialMovingAverage']))
#     print('sess.run(variables_to_restore) local3/biases/ExponentialMovingAverage',sess.run(variables_to_restore['local3/biases/ExponentialMovingAverage']))
#     print('sess.run(variables_to_restore) local4/weights/ExponentialMovingAverage',sess.run(variables_to_restore['local4/weights/ExponentialMovingAverage']))
#     print('sess.run(variables_to_restore) local4/biases/ExponentialMovingAverage',sess.run(variables_to_restore['local4/biases/ExponentialMovingAverage']))
#     print('sess.run(variables_to_restore) softmax_linear/weights/ExponentialMovingAverage',sess.run(variables_to_restore['softmax_linear/weights/ExponentialMovingAverage']))
#     print('sess.run(variables_to_restore) softmax_linear/biases/ExponentialMovingAverage',type(sess.run(variables_to_restore['softmax_linear/biases/ExponentialMovingAverage'])))


    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

#       num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))

      num_iter = FLAGS.max_iter_eval
      true_positives_count = 0
      false_positives_count = 0
      true_negatives_count = 0
      false_negatives_count = 0
      perfmetrics = {}
      
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():

        print('sess.run(labels)',sess.run(labels))

        is_label_one = sess.run(labels).astype(bool)
        is_label_zero = np.logical_not(is_label_one)
           
        correct_prediction = sess.run([top_k_op])
        false_prediction = np.logical_not(correct_prediction)
          
        true_positives_count += np.sum(np.logical_and(correct_prediction, is_label_one))
        false_positives_count += np.sum(np.logical_and(false_prediction, is_label_zero))
           
        true_negatives_count += np.sum(np.logical_and(correct_prediction, is_label_zero))
        false_negatives_count += np.sum(np.logical_and(false_prediction, is_label_one))     


#         predictions = sess.run([top_k_op])
#         true_count += np.sum(predictions)
#         print('predictions',predictions)
        step += 1
#         print('step',step)
#         print('correct_prediction',correct_prediction)
#         print('np.logical_and(correct_prediction,is_label_one)',np.logical_and(correct_prediction,is_label_one))
      
      # Compute precision @ 1.
#       precision = true_count / total_sample_count


      print('max_iter_eval',FLAGS.max_iter_eval)
      print('true_positives_count',true_positives_count)
      print('false_positives_count',false_positives_count)
      print('false_negatives_count',false_negatives_count)
      print('true_negatives_count',true_negatives_count)
      precision = float(true_positives_count) / float(true_positives_count+false_positives_count)
      recall = float(true_positives_count) / float(true_positives_count+false_negatives_count)
      f1score = 2*float(true_positives_count) / (2*float(true_positives_count)+float(false_positives_count + false_negatives_count))
      tpr = float(true_positives_count) / float(true_positives_count+false_negatives_count)
      fpr = float(false_positives_count) / float(false_positives_count+true_negatives_count)
      
      perfmetrics['precision'] = round(precision,FLAGS.numdecimalplaces)
      perfmetrics['recall'] = round(recall,FLAGS.numdecimalplaces)
      perfmetrics['f1score'] = round(f1score,FLAGS.numdecimalplaces)
      perfmetrics['tpr'] = round(tpr,FLAGS.numdecimalplaces)
      perfmetrics['fpr'] = round(fpr,FLAGS.numdecimalplaces)
      
#       print('logits',sess.run(logits))
#       print('labels',sess.run(labels))

      print('%s: classification training precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

#     return precision
    return perfmetrics

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)
    print('logits',logits)
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    if not FLAGS.out_dir:
      raise ValueError('Please supply a out_dir')
  
    
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      perfmetrics = eval_once(saver, summary_writer, top_k_op, summary_op,variables_to_restore,logits,labels)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
      
    return perfmetrics

def main(argv=None):  # pylint: disable=unused-argument
#   cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  perfmetrics = evaluate()


if __name__ == '__main__':
  tf.app.run()
