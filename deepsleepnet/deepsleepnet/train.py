#! /usr/bin/python
# -*- coding: utf8 -*-
import os

import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()

from deepsleep.trainer import DeepFeatureNetTrainer, DeepSleepNetTrainer
from deepsleep.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataFileList', 'data',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('dataset', 'data',
                           """Dataset to use. Supported datasets: shhs, wsc,"""
                           """cicc, isruc, mass, mros, SleepProfiler""")
tf.app.flags.DEFINE_string('output_dir', 'output',
                           """Directory where to save trained models """
                           """and outputs.""")
tf.app.flags.DEFINE_integer('n_folds', 20,
                           """Number of cross-validation folds.""")
tf.app.flags.DEFINE_integer('fold_idx', 0,
                            """Index of cross-validation fold to train.""")
tf.app.flags.DEFINE_integer('pretrain_epochs', 100,
                            """Number of epochs for pretraining DeepFeatureNet.""")
tf.app.flags.DEFINE_integer('finetune_epochs', 200,
                            """Number of epochs for fine-tuning DeepSleepNet.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Whether to resume the training process.""")


def pretrain(n_epochs):
    trainer = DeepFeatureNetTrainer(
        dataFileList=FLAGS.dataFileList,
        dataset=FLAGS.dataset,
        output_dir=FLAGS.output_dir,
        #n_folds=FLAGS.n_folds, 
        #fold_idx=FLAGS.fold_idx,
        batch_size=100, 
        input_dims=EPOCH_SEC_LEN*100, 
        n_classes=NUM_CLASSES,
        interval_plot_filter=50,
        interval_save_model=10,
        interval_print_cm=10
    )
    pretrained_model_path = trainer.train(
        n_epochs=n_epochs, 
        resume=FLAGS.resume
    )
    return pretrained_model_path


def finetune(model_path, n_epochs):
    trainer = DeepSleepNetTrainer(
        dataFileList=FLAGS.dataFileList,
        dataset=FLAGS.dataset,
        output_dir=FLAGS.output_dir, 
        #n_folds=FLAGS.n_folds, 
        #fold_idx=FLAGS.fold_idx, 
        batch_size=10, 
        input_dims=EPOCH_SEC_LEN*100, 
        n_classes=NUM_CLASSES,
        seq_length=25,
        n_rnn_layers=2,
        return_last=False,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    )
    finetuned_model_path = trainer.finetune(
        pretrained_model_path=model_path, 
        n_epochs=n_epochs, 
        resume=FLAGS.resume
    )
    return finetuned_model_path


def main(argv=None):
    # Output dir
    output_dir = FLAGS.output_dir #os.path.join(FLAGS.output_dir, "fold{}".format(FLAGS.fold_idx))
    if not FLAGS.resume:
        if tf.gfile.Exists(output_dir):
            tf.gfile.DeleteRecursively(output_dir)
        tf.gfile.MakeDirs(output_dir)

    pretrained_model_path = pretrain(
        n_epochs=FLAGS.pretrain_epochs
    )
    finetuned_model_path = finetune(
        model_path=pretrained_model_path, 
        n_epochs=FLAGS.finetune_epochs
    )
    return finetuned_model_path



def train(dataFileList,dataset,output_dir,pretrain_epochs = 100,finetune_epochs = 200,resume = False):
    #if 'finetuned_model_path' not in list(tf.compat.v1.flags.FLAGS): tf.app.flags.DEFINE_string('finetuned_model_path','','') # Using this as a way to obtain an output from app.run() until better solution is found
    FLAGS.dataFileList = dataFileList
    FLAGS.dataset = dataset
    FLAGS.output_dir = output_dir
    #FLAGS.n_folds = n_folds
    #FLAGS.fold_idx = fold_idx
    FLAGS.pretrain_epochs = pretrain_epochs
    FLAGS.finetune_epochs = finetune_epochs
    FLAGS.resume = resume
    #tf.compat.v1.app.run(main)
    return main()
