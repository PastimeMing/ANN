import os
import sys
import numpy as np
import tensorflow as tf

from MLP import MLP

from tensorflow.examples.tutorials.mnist import input_data

#Sets the threshold for what messages will be logged.
old_v = tf.logging.get_verbosity()
# able to set the logging verbosity to either DEBUG, INFO, WARN, ERROR, or FATAL. Here its ERROR
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
tf.logging.set_verbosity(old_v)

N_CLASSES = 10
BATCH_SIZE = 1000
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

hyper_param = {
    'learning_rate': 0.01,
    'keep_prob': 1.0,
    'weight_decay': WEIGHT_DECAY,
    'momentum': MOMENTUM,
    'state_switch':[1]*100
}

g = tf.Graph()
model = MLP('fully_model', N_CLASSES, BATCH_SIZE, g, [50,50])

for i in range(1,201):
    acc, loss = model.train_single_epoch(mnist, hyper_param)
    print("Train: ",i, " acc: ",acc," loss: ",loss)
    if i%100 == 0:
        result = model.test_single_epoch(mnist, hyper_param)
        print("Test: ",int(i/100), " acc: ",result )



# 保存模型
model.saver.save(model.sess, 'H:/tool\project\ANN/fully_model/Ann_model')
