#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:37:05 2017

@author: lakshay
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import numpy as np
import h5py
import tensorflow as tf
import time
from datetime import timedelta
import math
import random

import os, sys, pprint, time
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from model import *
from utils import *

import cv2
import errno
import pickle

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_string("dataset", "cifar10", "The name of dataset [cifar10, cifar100]")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("eval_step", 5, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 100, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_float("hyperparameter",0.7,"The parameter for scorer loss calculations")
flags.DEFINE_float("threshold",0.92,"The confidence for selecting a image")
FLAGS = flags.FLAGS

# Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

def prep_data():
    
    f = h5py.File('features_cifar10sorted.h5', 'r')
    
    train_features = np.array(f['train_images']).astype('float32') 
    #train_features = np.zeros((50000, 2048), dtype=np.float32)
    train_labels = np.array(f['train_labels'])
    train_files = np.array(f['train_files'])
   
    temp = []
    for i in range(train_labels.shape[0]):
        decoded = train_labels[i].decode('ASCII')
        temp.append(decoded)
    train_labels = np.array(temp)
    
    temp = []
    for i in range(train_files.shape[0]):
        decoded = train_files[i].decode('ASCII')
        temp.append(decoded)
    train_files = np.array(temp)
        
    return train_features, train_labels, train_files

###### ANN begins.

num_classes = 1

feature_size = 2048

fc_neurons = 1024

batch_size = 64
num_epochs = 24

eval_frequency = 10 # Number of steps between evaluations.

###  

def main(_): 
    pp.pprint(flags.FLAGS.__flags)
    
    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)
    
    with tf.device("/cpu:0"):
        ##========================= DEFINE MODEL ===========================##
        train_features, train_labels, train_files = prep_data()
        
        x_features = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, feature_size], name="features_batch")
        real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')

        # x --> scorer for training
        net_s, s_logits = sANN_simplified_api(x_features, is_train=True, reuse=False)
        # z --> generator for training
        z = tf.multiply(s_logits, x_features, name="noise_prod")
        net_g, g_logits = generator_simplified_api(z, is_train=True, reuse=False)
        # generated fake images --> discriminator
        net_d, d_logits = discriminator_simplified_api(net_g.outputs, is_train=True, reuse=False)
        # real images --> discriminator
        net_d2, d2_logits = discriminator_simplified_api(real_images, is_train=True, reuse=True)
        # sample_z --> generator for evaluation, set is_train to False
        # so that BatchNormLayer behave differently
        net_g2, g2_logits = generator_simplified_api(z, is_train=False, reuse=True)
        # so that scores can be generated after training
        net_s2, s2_logits = sANN_simplified_api(x_features, is_train=False, reuse=True)

        ##========================= DEFINE TRAIN OPS =======================##
        # cost for updating discriminator and generator
        # discriminator: real images are labelled as 1
        d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
        # discriminator: images from generator (fake) are labelled as 0
        d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
        d_loss = d_loss_real + d_loss_fake
        # generator: try to make the the fake images look real (1)
        g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')
        # cost for updating scorer
        s_loss=tf.subtract(tf.divide(tf.reduce_sum(s_logits), FLAGS.batch_size),FLAGS.hyperparameter)
        
        s_vars = tl.layers.get_variables_with_name('sANN', True, True)
        g_vars = tl.layers.get_variables_with_name('generator', True, True)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

        net_s.print_params(False)
        print("---------------")
        net_g.print_params(False)
        print("---------------")
        net_d.print_params(False)

        # optimizers for updating scorer, discriminator and generator
        d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(g_loss, var_list=g_vars)
        s_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(d_loss, var_list=s_vars)
        
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)
    tl.files.exists_or_mkdir(save_dir)
    # load the latest checkpoints
    net_s_name = os.path.join(save_dir, 'net_s.npz')
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    #sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float64)# sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.sample_size, z_dim)).astype(np.float32)

    if FLAGS.is_train == False:
        print("Prediction Mode ON!")
        total_files_eval = (min(len(train_files), FLAGS.train_size) // FLAGS.batch_size) * FLAGS.batch_size
        tl.files.load_and_assign_npz(sess, name=net_s_name, network=net_s2)
        batch_idxs = min(len(train_files), FLAGS.train_size) // FLAGS.batch_size
        score_idx = 0
        count_dict={}
        for idx in range(batch_idxs):
            batch_features = train_features[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            batch_score = sess.run([net_s2.outputs], feed_dict={x_features : batch_features})
            print(len(batch_score))
	    print(batch_score)
            for bidx in range(FLAGS.batch_size):
                if(batch_score[0][bidx] > FLAGS.threshold):
                    label_of_image=train_labels[score_idx]
                    if (label_of_image not in count_dict):
                        count_dict[label_of_image]=1
                    else:
                        count_dict[label_of_image]+=1
                score_idx = score_idx + 1
        print(count_dict)
        exit()
    
    ##========================= TRAIN MODELS ================================##
    total_files_eval = (min(len(train_files), FLAGS.train_size) // FLAGS.batch_size) * FLAGS.batch_size
    num_evals = (FLAGS.epoch // FLAGS.eval_step)
    result_scores = np.zeros((num_evals, total_files_eval), dtype=np.float32)
    res_idx = 0
    for epoch in range(FLAGS.epoch):

        ## load image data
        batch_idxs = min(len(train_files), FLAGS.train_size) // FLAGS.batch_size

        for idx in range(batch_idxs):
            batch_files = train_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            ## get real images
            batch_images = []
            for fl in batch_files:
                image = cv2.imread(fl)
                image = cv2.resize(image, (FLAGS.output_size, FLAGS.output_size), 0, 0, cv2.INTER_CUBIC)
                image = image.astype(np.float32)
                batch_images.append(image)
            batch_images = np.array(batch_images)
            batch_features = train_features[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            start_time = time.time()
            # updates the discriminator
            errD, _ = sess.run([d_loss, d_optim], feed_dict={x_features: batch_features, real_images: batch_images })
            # updates the generator, run generator twice to make sure that d_loss does not go to zero (difference from paper)
            for _ in range(2):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={x_features: batch_features, real_images: batch_images })
            # updates the scorer
            errS, _ = sess.run([s_loss, s_optim], feed_dict={x_features: batch_features, real_images: batch_images })
            #print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            #        % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time, errD, errG))
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f. s_loss: %.8f" \
                    % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time, errD, errG, errS))

        if np.mod((epoch + 1), FLAGS.eval_step) == 0:
            # save current network parameters
            print("[*] Saving checkpoints...")
            tl.files.save_npz(net_s.all_params, name=net_s_name, sess=sess)
            tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
            tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
            print("[*] Saving checkpoints SUCCESS!")
            # generate the list of selected images.
            batch_idxs = min(len(train_files), FLAGS.train_size) // FLAGS.batch_size
            score_idx = 0
            for idx in range(batch_idxs):
                batch_features = train_features[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                batch_score, errS = sess.run([net_s2.outputs, s_loss], feed_dict={x_features : batch_features})
                for bidx in range(FLAGS.batch_size):
                    result_scores[res_idx][score_idx] = batch_score[bidx][0]
                    score_idx = score_idx + 1
            print("[Evaluation] s_loss: %.8f" % (errS))
            res_idx = res_idx + 1
    
    ##################### Understanding the scores ##################
    num_of_epochs_recorded = result_scores.shape[0]
    num_of_files_recorded = result_scores.shape[1]
    main_dict={} #dictionary of dictionaries of type {epoch0:{"airplane":100,"frog":10...},epoch1:{"airplane":...}...}
    for i in range(num_of_epochs_recorded):
        count_dict={}#dictionary for each epoch containing the count of each label
        for j in range(num_of_files_recorded):
            ##### Check if score greater than threshold.... if yes... then for that j, see the label from train_labels and increase the count of that label
            if(result_scores[i][j]>FLAGS.threshold):
                label_of_image=train_labels[j]
                if (label_of_image not in count_dict):
                    count_dict[label_of_image]=1
                else:
                    count_dict[label_of_image]+=1
        main_dict[i]=count_dict
    
    for key in main_dict:
        print(key, " => ", main_dict[key])
    
    try:
        os.makedirs("./Outputs/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    outputFileName = "./Outputs/output" + time.strftime("%Y%m%d-%H%M%S") + ".pickle"
    pickle_out = open(outputFileName, "wb")
    pickle.dump(main_dict, pickle_out)
    pickle_out.close()
    
if __name__ == '__main__':
    tf.app.run()


