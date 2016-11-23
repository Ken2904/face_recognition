#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2
import os
import math

face_class=np.load("text_npy/face_class.npy")

NUM_CLASSES = len(face_class)
IMAGE_SIZE = 96
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

def inference(images_placeholder, keep_prob):
    """ 予測モデルを作成する関数

    引数: 
      images_placeholder: 画像のplaceholder
      keep_prob: dropout率のplace_holder

    返り値:
      y_conv: 各クラスの確率(のようなもの)
    """
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 畳み込み層の作成
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層の作成
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
      #正規化層の追加                      
    def normalization(x, dim):
    	return tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None) 
    
    # 入力を28x28x3に変形
    x_image = tf.reshape(images_placeholder, [-1, 96, 96, 3])

    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
        
	# 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
    	W_conv2 = weight_variable([5, 5, 32, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
    
    # 畳み込み層3の作成
    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([5, 5, 32, 32])
        b_conv3 = bias_variable([32])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # プーリング層2の作成
    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool_2x2(h_conv3)
        
    with tf.name_scope('norm1') as scope:
    	h_norm1= normalization(h_pool3, 2)

    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([12*12*32, 4000])
        b_fc1 = bias_variable([4000])
        h_norm1_flat = tf.reshape(h_norm1, [-1, 12*12*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_norm1_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([4000, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv
