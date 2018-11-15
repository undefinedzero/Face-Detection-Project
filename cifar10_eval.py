from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf

import cifar10

parser = cifar10.parser

parser.add_argument('--eval_dir', type=str, default='./cifar10_eval',
                    help='Directory where to write event logs.')

parser.add_argument('--eval_data', type=str, default='test',
                    help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str, default='./cifar10_train',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=60*5,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=20976, # 测试集的图片张数
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=True,  # 只跑一次，设置成True
                    help='Whether to run eval only once.')


def eval_once(saver, summary_writer, top_k_op, summary_op, predict_label): # 把predict_label作为参数传入
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

    # Start the queue runners.
    coord = tf.train.Coordinator()

    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      output = [] # output用来保存最终预测的label
      while step < num_iter and not coord.should_stop():
        # ----------------------------------------------
        predictions, predict_label_ = sess.run([top_k_op, predict_label]) # run，来输出label
        # ----------------------------------------------
        true_count += np.sum(predictions)
        step += 1
        print(predict_label_) # 输出来看看
        # ----------------------------------------------
        output.append(predict_label_) # 把每一个batch 预测的label append到output
        # ----------------------------------------------

      # ----------------------------------------------
      save_path = r'./cifar10_train'
      np.savetxt(os.path.join(save_path, "predict_label.txt"), output, fmt="%d", delimiter="\n") # 以列向量的形式保存，注意之前要import os
      # ----------------------------------------------

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)


    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    # ----------------------------------------------
    predict_label = tf.argmax(logits, axis=1) # 预测的label是logits中最大的那个值所在的下标
    # ----------------------------------------------

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      # ----------------------------------------------
      eval_once(saver, summary_writer, top_k_op, summary_op, predict_label) # 把predict_label作为参数传入
      # ----------------------------------------------
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    pass
  else:
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()
