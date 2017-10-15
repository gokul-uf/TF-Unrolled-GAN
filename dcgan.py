from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


class DCGAN(object):
    """
        Tensorflow implementation of DCGAN, with four CNN layers.
        We assume the input images are of size 32x32.
    """

    def __init__(self):
        # self.image_size = 64
        self.image_size = 32
        self.noise_size = 100
        self.lrelu_alpha = 0.2
        self.num_channels = 3
        self.lr = 0.0002
        self.beta_1 = 0.5

    def _create_placeholders(self):
        self.input_images = tf.placeholder(
            shape=[None, self.image_size, self.image_size, self.num_channels],
            dtype=tf.float32,
            name="input_images")
        self.input_noise = tf.placeholder(
            shape=[None, self.noise_size],
            dtype=tf.float32,
            name="input_noise")

    def _create_generator(self):
        xav_init = tf.contrib.layers.xavier_initializer
        bnorm = tf.layers.batch_normalization
        with tf.variable_scope("generator"):
            """
            fc_1 = tf.layers.dense(
                inputs=self.input_noise, units=4 * 4 * 512, name="fc_1")
            """

            fc_1 = tf.layers.dense(
                inputs=self.input_noise, units=4 * 4 * 256, name="fc_1")
            reshaped_fc_1 = tf.reshape(
                fc_1,
                shape=[tf.shape(fc_1)[0], 4, 4, 256],
                name="reshapsed_noise")

            def _create_deconv_bnorm_block(inputs,
                                           name,
                                           filters,
                                           activation=tf.nn.relu):
                with tf.variable_scope(name):
                    deconv = tf.layers.conv2d_transpose(
                        inputs=inputs,
                        filters=filters,
                        kernel_size=[5, 5],
                        strides=2,
                        padding="same",
                        kernel_initializer=xav_init(),
                        name="deconv")
                    deconv = activation(deconv)
                    bnorm_op = bnorm(deconv, name="bnorm")
                    return bnorm_op

            """
            bnorm_1 = _create_deconv_bnorm_block(
                inputs=reshaped_fc_1, filters=256, name="block_1")
            bnorm_2 = _create_deconv_bnorm_block(
                inputs=bnorm_1, filters=128, name="block_2")
            """

            bnorm_2 = _create_deconv_bnorm_block(
                inputs=reshaped_fc_1, filters=128, name="block_2")
            bnorm_3 = _create_deconv_bnorm_block(
                inputs=bnorm_2, filters=64, name="block_3")

            bnorm_4 = _create_deconv_bnorm_block(
                inputs=bnorm_3,
                filters=3,
                activation=tf.nn.tanh,
                name="block_4")

            return bnorm_4

    def _create_discriminator(self, inputs, reuse=False):
        xav_init = tf.contrib.layers.xavier_initializer
        bnorm = tf.layers.batch_normalization
        with tf.variable_scope("discriminator", reuse=reuse):

            def _create_conv_bnorm_block(inputs, filters, name):
                with tf.variable_scope(name, reuse=reuse):
                    conv = tf.layers.conv2d(
                        inputs=inputs,
                        filters=filters,
                        kernel_size=[5, 5],
                        strides=2,
                        padding="same",
                        kernel_initializer=xav_init(),
                        name="conv")

                    conv = tf.maximum(conv, self.lrelu_alpha * conv)
                    bnorm_op = bnorm(conv, name="bnorm")
                    return bnorm_op

            conv_1 = tf.layers.conv2d(
                inputs=inputs,
                filters=64,
                kernel_size=[5, 5],
                strides=2,
                kernel_initializer=xav_init(),
                padding="same",
                name="conv_1")
            conv_1 = tf.maximum(conv_1, self.lrelu_alpha * conv_1)

            bnorm_1 = _create_conv_bnorm_block(
                inputs=conv_1, filters=128, name="block_1")

            bnorm_2 = _create_conv_bnorm_block(
                inputs=bnorm_1, filters=256, name="block_2")
            """
            bnorm_3 = _create_conv_bnorm_block(
                inputs=bnorm_2, filters=512, name="block_3")

            reshaped_bnorm_3 = tf.reshape(
                bnorm_3,
                shape=[tf.shape(bnorm_3)[0], 4 * 4 * 512],
                name="reshaped_bnorm_3")

            logits = tf.layers.dense(
                inputs=reshaped_bnorm_3, units=1, name="fc_1")
            """
            reshaped_bnorm_2 = tf.reshape(
                bnorm_2,
                shape=[tf.shape(bnorm_2)[0], 4 * 4 * 256],
                name="reshaped_bnorm_2")

            logits = tf.layers.dense(
                inputs=reshaped_bnorm_2, units=1, name="fc_1")
            fc_1 = tf.sigmoid(logits)
            return fc_1, logits

    def _compute_loss(self):
        self.d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.real_logits, labels=tf.ones_like(self.real_logits))
        self.d_loss_real = tf.reduce_mean(self.d_loss_real)

        self.d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.fake_logits, labels=tf.zeros_like(self.fake_logits))
        self.d_loss_fake = tf.reduce_mean(self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.fake_logits, labels=tf.ones_like(self.fake_logits))
        self.g_loss = tf.reduce_mean(self.g_loss)

        tf.summary.scalar("disc_loss_real", self.d_loss_real)
        tf.summary.scalar("disc_loss_fake", self.d_loss_fake)
        tf.summary.scalar("disc_loss", self.d_loss)
        tf.summary.scalar("gen_loss", self.g_loss)

        d_opt = tf.train.AdamOptimizer(
            learning_rate=self.lr, beta1=self.beta_1)
        g_opt = tf.train.AdamOptimizer(
            learning_rate=self.lr, beta1=self.beta_1)

        d_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
        g_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")

        self.d_train = d_opt.minimize(self.d_loss, var_list=d_vars)
        self.g_train = g_opt.minimize(self.g_loss, var_list=g_vars)
