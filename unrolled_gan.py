from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from dcgan import DCGAN
import tensorflow as tf

Adam = tf.contrib.keras.optimizers.Adam
_graph_replace = tf.contrib.graph_editor.graph_replace


class unrolled_gan(DCGAN):
    def __init__(self, unroll_steps):
        super(unrolled_gan, self).__init__()
        self.unroll_steps = unroll_steps
        self._create_placeholders()
        self.generator_output = self._create_generator()
        self.real_predictions, self.real_logits = self._create_discriminator(
            inputs=self.input_images)
        self.fake_predictions, self.fake_logits = self._create_discriminator(
            inputs=self.generator_output, reuse=True)
        self._compute_loss()
        self.summary_op = tf.summary.merge_all()

    def remove_original_op_attributes(self, graph):
        for op in graph.get_operations():
            op._original_op = None

    def graph_replace(self, *args, **kwargs):
        self.remove_original_op_attributes(tf.get_default_graph())
        return _graph_replace(*args, **kwargs)

    def extract_update_dict(self, update_ops):
        name_to_var = {v.name: v for v in tf.global_variables()}
        updates = OrderedDict()
        for update in update_ops:
            var_name = update.op.inputs[0].name
            var = name_to_var[var_name]
            value = update.op.inputs[1]
            if update.op.type == 'Assign':
                updates[var.value()] = value
            elif update.op.type == 'AssignAdd':
                updates[var.value()] = var + value
            else:
                raise ValueError(
                    "Update op type (%s) must be of type Assign or AssignAdd" %
                    update_op.op.type)
        return updates

    def _compute_loss(self):
        self.d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.real_logits, labels=tf.ones_like(self.real_logits))
        self.d_loss_real = tf.reduce_mean(self.d_loss_real)

        self.d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.fake_logits, labels=tf.zeros_like(self.fake_logits))
        self.d_loss_fake = tf.reduce_mean(self.d_loss_fake)

        # non-unrolled disc_loss
        self.d_loss = self.d_loss_real + self.d_loss_fake

        g_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
        d_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

        # We use the same hparams as DCGAN coz it's stable
        d_opt = Adam(lr=self.lr, beta_1=self.beta_1)
        updates = d_opt.get_updates(d_vars, [], self.d_loss)
        self.d_train_op = tf.group(*updates, name="d_train_op")

        if self.unroll_steps > 0:
            update_dict = self.extract_update_dict(updates)
            cur_update_dict = update_dict
            for i in range(self.unroll_steps - 1):
                cur_update_dict = self.graph_replace(update_dict,
                                                     cur_update_dict)
            self.unrolled_loss = self.graph_replace(self.d_loss,
                                                    cur_update_dict)
        else:
            self.unrolled_loss = self.d_loss

        g_opt = tf.train.AdamOptimizer(
            learning_rate=self.lr, beta1=self.beta_1)
        self.g_train = g_opt.minimize(-self.unrolled_loss, var_list=g_vars)

        # self.d_train = d_opt.minimize(self.d_loss, var_list=d_vars)
        # self.g_train = g_opt.minimize(self.g_loss, var_list=g_vars)

        tf.summary.scalar("disc_loss_real", self.d_loss_real)
        tf.summary.scalar("disc_loss_fake", self.d_loss_fake)
        tf.summary.scalar("disc_loss", self.d_loss)
        tf.summary.scalar("unrolled_loss", self.unrolled_loss)
        # tf.summary.scalar("gen_loss", self.g_loss)
