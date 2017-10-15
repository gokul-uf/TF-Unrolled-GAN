from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
import tensorflow as tf

from model import DCGAN
from data_utils import Processor

flags = tf.app.flags
FLAGS = flags.FLAGS

# Else, you won't see anything in stderr with --alsologtostderr
tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string("data_dir", None, "The location of the dataset")
flags.DEFINE_string("output_dir", None, "Where should the outputs be stored")
flags.DEFINE_integer("save_every", 500, "Save checkpoints every N steps")
flags.DEFINE_integer("eval_every", 50, "Generate images every N steps")
flags.DEFINE_integer("eval_images", 100,
                     "Images to generate at eval, must be a perfect square")
flags.DEFINE_integer("num_steps", 1000, "Number of batchs to train on")
flags.DEFINE_integer("batch_size", 100, "Batch size")


def maybe_create_output_dir():
    if os.path.exists(FLAGS.output_dir):
        tf.logging.info("data_dir already exists, not creating again")
        return

    os.mkdir(FLAGS.output_dir)
    os.mkdir(os.path.join(FLAGS.output_dir, "ckpts"))
    os.mkdir(os.path.join(FLAGS.output_dir, "summaries"))
    os.mkdir(os.path.join(FLAGS.output_dir, "images"))
    tf.logging.info("All paths created!")


def create_collage(images, step):
    n_cols = np.sqrt(FLAGS.eval_images).astype(np.int32)
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(n_cols, n_cols), axes_pad=0)
    for i in range(FLAGS.eval_images):
        grid[i].imshow(images[i])
        grid[i].set_xticks([])
        grid[i].set_yticks([])
    plt.savefig(
        os.path.join(FLAGS.output_dir, "images", "step_{}.png".format(step)))
    plt.close()
    tf.logging.info("Saved generated images")


if __name__ == "__main__":

    tf.logging.info("Starting training for %d steps", FLAGS.num_steps)
    tf.logging.info("Passed flags: %s", FLAGS.__flags)
    dcgan = DCGAN()
    processor = Processor(data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
    data_yielder = processor.get_batch()
    saver = tf.train.Saver(max_to_keep=None)
    maybe_create_output_dir()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(
            os.path.join(FLAGS.output_dir, "summaries"), sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.num_steps):
            train_batch = data_yielder.next()

            # len(...) because we can get smaller batches at file edges
            noise = np.random.randn(len(train_batch), dcgan.noise_size)

            fetches = [
                dcgan.d_train, dcgan.g_train, dcgan.d_loss, dcgan.g_loss,
                dcgan.summary_op
            ]
            feed_dict = {
                dcgan.input_images: train_batch,
                dcgan.input_noise: noise
            }
            _, _, d_loss, g_loss, summary = sess.run(
                fetches, feed_dict=feed_dict)

            tf.logging.log_every_n(tf.logging.INFO,
                                   "Step {}, G Loss: {}, D Loss: {}".format(
                                       i, g_loss, d_loss), FLAGS.eval_every)

            if i % FLAGS.eval_every == 0:
                # Let's generate some images!
                tf.logging.info("Running evaluation")
                feed_dict = {
                    dcgan.input_noise:
                    np.random.randn(FLAGS.eval_images, dcgan.noise_size)
                }
                gen_output = sess.run(
                    dcgan.generator_output, feed_dict=feed_dict)
                gen_output = (gen_output * 127) + 127.0
                gen_output = gen_output.astype(np.uint8)
                create_collage(gen_output, i)
                summary_writer.add_summary(summary, i)

            if i % FLAGS.save_every == 0:
                # Save the trained model
                filename = saver.save(
                    sess,
                    os.path.join(FLAGS.output_dir,
                                 "ckpts/step_{}.ckpt".format(i)))
                tf.logging.info(
                    "Saved trained model after %d steps with filename %s", i,
                    filename)
