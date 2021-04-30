import os.path
import time

import tensorflow as tf

from data_utils import Vocabulary, Dataset
from language_model import LM
from run_utils import run_train, run_eval


flags = tf.flags
flags.DEFINE_string('logdir', None, 'Logging directory.')
flags.DEFINE_string('datadir', "/home/gpzlx1/mylm1b/dataset/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*", 'Data directory.')
flags.DEFINE_string('vocab', "1b_word_vocab.txt", 'Vocab file.')
flags.DEFINE_string(
    'mode', 'train',
    'Whether to run "train" or "eval_(train|valid|test)" model.')
flags.DEFINE_string('hpconfig', '', 'Overrides default hyper-parameters.')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs used.')
flags.DEFINE_integer('eval_steps', 70, 'Number of eval steps.')

FLAGS = flags.FLAGS


def main(_):
    hps = LM.get_default_hparams().parse(FLAGS.hpconfig)
    hps.num_gpus = FLAGS.num_gpus

    vocab = Vocabulary.from_file(FLAGS.vocab)
    hps.vocab_size = vocab.num_tokens


    if FLAGS.logdir is None:
        FLAGS.logdir = os.path.join('/tmp', 'lm-run-{}'.format(int(time.time())))
        print('logdir: {}'.format(FLAGS.logdir))
    hps.batch_size = 256
    dataset = Dataset(vocab, FLAGS.datadir)
    run_train(dataset, hps, FLAGS.logdir + '/train', ps_device='/gpu:0')
 


if __name__ == '__main__':
    tf.app.run()
