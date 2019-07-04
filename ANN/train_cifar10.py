import os
import sys
import numpy as np
import tensorflow as tf
from MLP_cifar import MLP
from data_providers.utils import get_data_provider_by_name

N_CLASSES = 10
BATCH_SIZE = 10
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

train_params = {
    'batch_size': BATCH_SIZE,
    'validation_set': False,
    'shuffle': 'every_epoch',
    'normalization': 'by_chanels',
    'save_path': os.path.join('.', 'data', 'cifar')
}
data_provider = get_data_provider_by_name('C10', train_params)

#Sets the threshold for what messages will be logged.
old_v = tf.logging.get_verbosity()
# able to set the logging verbosity to either DEBUG, INFO, WARN, ERROR, or FATAL. Here its ERROR
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(old_v)


hyper_param = {
    'learning_rate': 0.01,
    'keep_prob': 1.0,
    'weight_decay': WEIGHT_DECAY,
    'momentum': MOMENTUM,
    'state_switch':[1]*50
}

g = tf.Graph()
model = MLP('fully_model_cifar', N_CLASSES, BATCH_SIZE, g, [50])

for i in range(1,201):
    acc, loss = model.train_single_epoch(data_provider, hyper_param)
    print("Train: ",i, " acc: ",acc," loss: ",loss)
    # if i%100 == 0:
    result = model.test_single_epoch(data_provider, hyper_param)
    print("Test: ",int(i/100), " acc: ",result )



# 保存模型
model.saver.save(model.sess, 'H:/tool\project\ANN/fully_model/Ann_cifar10_model')
