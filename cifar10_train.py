from resnet import *
import tensorflow as tf
from datetime import datetime
import time
from cifar10_input import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('version', 'res_noReg', '''Path to the CIFAR-10 data directory''')
tf.app.flags.DEFINE_float('init_lr', 0.1, '''initial learning rate''')
tf.app.flags.DEFINE_boolean('is_data_augmentation', True, '''If use the padding-cropping
augmentation''')
tf.app.flags.DEFINE_boolean('is_advanced_augmentation', False, '''If use distortion and
brightening etc.''')
tf.app.flags.DEFINE_integer('num_residual_blocks', 3, '''How many residual blocks do you want''')

ckpt_path = 'cache/logs_repeat20/model.ckpt-100000'
is_use_ckpt = False

# Total steps to train
STEPS = 64000
# How many steps to generate one line of verbose?
REPORT_FREQ = 391
# Total layer = NUM_RESIDUAL_BLOCKS * 6 + 2
NUM_RESIDUAL_BLOCKS = FLAGS.num_residual_blocks
TRAIN_BATCH_SIZE = 128
VALI_BATCH_SIZE = 128
train_dir = 'logs_' + FLAGS.version + '/'

LR_DECAY_FACTOR = 0.1
TRAIN_EMA_DECAY = 0.95

DECAY_STEP0 = 32000
DECAY_STEP1 = 48000


class Train:
    def __init__(self):
        self.placeholders()
        self.main()

    def loss(self, logits, labels):

        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def top_k_error(self, predictions, labels, k):

        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)

        return (batch_size - num_correct) / float(batch_size)


    def placeholders(self):
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[TRAIN_BATCH_SIZE,
                                                                        IMG_HEIGHT,
                                                                    IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[TRAIN_BATCH_SIZE])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[VALI_BATCH_SIZE,
                                                                IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[VALI_BATCH_SIZE])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    def train_operation(self, global_step, total_loss, top1_error):

        tf.scalar_summary('learning_rate', self.lr_placeholder)
        tf.scalar_summary('train_loss', total_loss)
        tf.scalar_summary('train_top1_error', top1_error)

        ema = tf.train.ExponentialMovingAverage(TRAIN_EMA_DECAY, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.scalar_summary('train_top1_error_avg', ema.average(top1_error))
        tf.scalar_summary('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op

    def validation_op(self, validation_step, top1_error, loss):
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)
        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)
        tf.scalar_summary('val_top1_error', top1_error_val)
        tf.scalar_summary('val_top1_error_avg', top1_error_avg)
        tf.scalar_summary('val_loss', loss_val)
        tf.scalar_summary('val_loss_avg', loss_val_avg)
        return val_op

    def train(self):
        all_data, all_labels = prepare_train_data()
        vali_data, vali_labels = read_validation_data()

        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        if FLAGS.is_data_augmentation is True:
            paddings = [[0, 0], [4, 4], [4, 4], [0, 0]]
            augmented_image = tf.pad(self.image_placeholder,paddings=paddings, mode='CONSTANT')
            input_image_tensor = tf.random_crop(augmented_image, size=[TRAIN_BATCH_SIZE,
                                                                       IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH])
            if FLAGS.is_advanced_augmentation is True:
                input_image_tensor = tf.map_fn(tf.image.random_flip_left_right, input_image_tensor)
                fn = lambda x: tf.image.random_brightness(x, max_delta=63)
                input_image_tensor = tf.map_fn(fn, input_image_tensor)
                fn = lambda x: tf.image.random_contrast(x, lower=0.2, upper=1.8)
                input_image_tensor = tf.map_fn(fn, input_image_tensor)
            logits = inference_small(input_image_tensor, NUM_RESIDUAL_BLOCKS, reuse=False)
        else:
            logits = inference_small(self.image_placeholder, NUM_RESIDUAL_BLOCKS,
                                            reuse=False)
        vali_logits = inference_small(self.vali_image_placeholder, NUM_RESIDUAL_BLOCKS,
                                        reuse=True)


        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        full_loss = tf.add_n([loss] + reg_losses)
        predictions = tf.nn.softmax(logits)
        top1_error = self.top_k_error(predictions, self.label_placeholder, 1)

        vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)


        train_op, train_ema_op = self.train_operation(global_step, full_loss, top1_error)
        val_op = self.validation_op(validation_step, vali_top1_error, vali_loss)

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        sess = tf.Session()

        if is_use_ckpt is True:
            saver.restore(sess, ckpt_path)
            print 'Restored from checkpoint...'
        else:
            sess.run(init)

        summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

        for step in xrange(STEPS):

                offset = np.random.choice(EPOCH_SIZE - TRAIN_BATCH_SIZE, 1)

                batch_data = all_data[offset:offset+TRAIN_BATCH_SIZE, ...]
                batch_label = all_labels[offset:offset+TRAIN_BATCH_SIZE]

                vali_image_batch, vali_labels_batch = generate_vali_batch(vali_data,
                                                               vali_labels, VALI_BATCH_SIZE)

                start_time = time.time()

                if step == 0:
                    _, top1_error_value, vali_loss_value = sess.run([val_op, vali_top1_error,
                                                                     vali_loss],
                                                    {self.image_placeholder: batch_data,
                                                     self.label_placeholder: batch_label,
                                                     self.vali_image_placeholder: vali_image_batch,
                                                     self.vali_label_placeholder: vali_labels_batch,
                                                     self.lr_placeholder: FLAGS.init_lr})
                    print 'Validation top1 error %.4f' % top1_error_value
                    print 'Validation loss = ', vali_loss_value
                    print '----------------------------'


                _, _, loss_value, train_top1_error = sess.run([train_op, train_ema_op, loss,
                        top1_error], {self.image_placeholder: batch_data,
                                      self.label_placeholder: batch_label,
                                      self.vali_image_placeholder: vali_image_batch,
                                      self.vali_label_placeholder: vali_labels_batch,
                                      self.lr_placeholder: FLAGS.init_lr})
                duration = time.time() - start_time

                if step % REPORT_FREQ == 0:
                    summary_str = sess.run(summary_op, {self.image_placeholder: batch_data,
                                                        self.label_placeholder: batch_label,
                                                        self.vali_image_placeholder: vali_image_batch,
                                                        self.vali_label_placeholder: vali_labels_batch,
                                                        self.lr_placeholder: FLAGS.init_lr})
                    summary_writer.add_summary(summary_str, step)


                    num_examples_per_step = TRAIN_BATCH_SIZE
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                    print format_str % (datetime.now(), step, loss_value, examples_per_sec,
                                       sec_per_batch)
                    print 'Train top1 error = ', train_top1_error


                    _, top1_error_value, vali_loss_value = sess.run([val_op, vali_top1_error,
                                                                     vali_loss],
                                                    {self.image_placeholder: batch_data,
                                                     self.label_placeholder: batch_label,
                                                     self.vali_image_placeholder: vali_image_batch,
                                                     self.vali_label_placeholder: vali_labels_batch,
                                                     self.lr_placeholder: FLAGS.init_lr})
                    print 'Validation top1 error %.4f' % top1_error_value
                    print 'Validation loss = ', vali_loss_value
                    print '----------------------------'


                if step == DECAY_STEP0 or step == DECAY_STEP1:
                    FLAGS.init_lr = 0.1 * FLAGS.init_lr
                    print 'Learning rate decayed to ', FLAGS.init_lr

                if step % 10000 == 0 or (step + 1) == STEPS:
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

    def main(self):
        self.train()

maybe_download_and_extract()
train = Train()
