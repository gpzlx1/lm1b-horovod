import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from language_model import LM
from common import CheckpointLoader
import horovod.tensorflow as hvd

def run_train(dataset, hps, logdir, ps_device, task=0, master=''):
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=2,
                            inter_op_parallelism_threads=20)
    
    with tf.variable_scope('model'):
        model = LM(hps, 'train', ps_device)

    print('ALL VARIABLES')
    for v in tf.all_variables():
        print('%s %s %s' % (v.name, v.get_shape(), v.device))
    print('TRAINABLE VARIABLES')
    for v in tf.trainable_variables():
        print('%s %s %s' % (v.name, v.get_shape(), v.device))
    print('LOCAL VARIABLES')
    for v in tf.local_variables():
        print('%s %s %s' % (v.name, v.get_shape(), v.device))

    

    #sv = tf.train.Supervisor(
    #    is_chief=(task == 0),
    #    logdir=logdir,
    #    summary_op=None,  # Automatic summaries don't work with placeholders.
    #    global_step=model.global_step,
    #    save_summaries_secs=30,
    #    save_model_secs=120 * 5)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    total_step = 0
    with tf.train.MonitoredTrainingSession(config=config, hooks=hooks) as sess:
        for v in tf.get_collection('initial_state'):
            sess.run(v.initializer, feed_dict={model.batch_size: hps.batch_size})
        # Slowly increase the number of workers during
        # beginning of the training.
        while not sess.should_stop():
            step = int(sess.run(model.global_step))
            waiting_until_step = task * hps.num_delayed_steps
            if step >= waiting_until_step:
                break
            else:
                print('Current step is %d. Waiting until: %d' %
                      (step, waiting_until_step))
            time.sleep(10.0)

        local_step = 0
        prev_global_step = sess.run(model.global_step)
        prev_time = time.time()
        data_iterator = dataset.iterate_forever(
            hps.batch_size * hps.num_gpus, hps.num_steps)
        while not sess.should_stop():
            fetches = [model.global_step, model.loss, model.train_op]
            # Chief worker computes summaries every 20 steps.
            should_compute_summary = (
                hvd.rank() == 0 and local_step > 0 and local_step % 20 == 0)
            if should_compute_summary:
                fetches += [model.summary_op]

            x, y, w = next(data_iterator)
            fetched = sess.run(fetches, {model.x: x, model.y: y, model.w: w})

            local_step += 1
            #if should_compute_summary:
            #    sess.summary_computed(sess, fetched[-1])
            if hvd.rank() == 0:
                if local_step < 10 or local_step % 200 == 0:
                    cur_time = time.time()
                    num_words = hps.batch_size * hps.num_gpus * hps.num_steps
                    sps = hps.batch_size * hps.num_gpus * (fetched[0] - prev_global_step) / (cur_time - prev_time)
                    wps = ((fetched[0] - prev_global_step) * num_words /
                           (cur_time - prev_time))
                    prev_global_step = fetched[0]
                    print('Iteration %d, time = %.2fs, wps = %.0f, sps = %.0f '
                          'train loss = %.4f' % (
                            fetched[0], cur_time - prev_time, wps * hvd.size(), sps * hvd.size(), fetched[1]))
                    prev_time = cur_time
    #sv.stop()


def run_eval(dataset, hps, logdir, mode, num_eval_steps):
    with tf.variable_scope('model'):
        hps.num_sampled = 0  # Always using full softmax at evaluation.
        hps.keep_prob = 1.0
        model = LM(hps, 'eval', '/cpu:0')

    if hps.average_params:
        print('Averaging parameters for evaluation.')
        saver = tf.train.Saver(model.avg_dict)
    else:
        saver = tf.train.Saver()

    # Use only 4 threads for the evaluation.
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=1)
    sess = tf.Session(config=config)
    sw = tf.summary.FileWriter(logdir + '/' + mode, sess.graph)
    ckpt_loader = CheckpointLoader(saver, model.global_step, logdir + '/train')

    with sess.as_default():
        while ckpt_loader.load_checkpoint():
            global_step = ckpt_loader.last_global_step
            data_iterator = dataset.iterate_once(
                hps.batch_size * hps.num_gpus, hps.num_steps)
            tf.initialize_local_variables().run()
            for v in tf.get_collection('initial_state'):
                sess.run(v.initializer,
                         feed_dict={model.batch_size: hps.batch_size})
            loss_nom = 0.0
            loss_den = 0.0
            for i, (x, y, w) in enumerate(data_iterator):
                if i >= num_eval_steps:
                    break

                loss = sess.run(model.loss, {
                    model.x: x, model.y: y, model.w: w,
                    model.batch_size: hps.batch_size})
                loss_nom += loss
                loss_den += w.mean()
                loss = loss_nom / loss_den
                sys.stdout.write('%d: %.3f (%.3f) ... ' % (
                    i, loss, np.exp(loss)))
                sys.stdout.flush()
            sys.stdout.write('\n')

            log_perplexity = loss_nom / loss_den
            print('Results at %d: log_perplexity = %.3f perplexity = %.3f' % (
                global_step, log_perplexity, np.exp(log_perplexity)))

            summary = tf.Summary()
            summary.value.add(
                tag='eval/log_perplexity', simple_value=log_perplexity)
            summary.value.add(
                tag='eval/perplexity', simple_value=np.exp(log_perplexity))
            sw.add_summary(summary, global_step)
            sw.flush()
