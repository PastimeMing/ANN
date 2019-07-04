import os
import numpy as np
import tensorflow as tf

class MLP:
    def __init__(self, name, n_classes, batch_size, graph, layer_infor=[]):
        self.name = name
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.layer_infor = layer_infor
        self.data_shape = [784]
        self.graph = graph

        self._define_inputs()
        self._build_graph()
        self._init_session()

    
    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(tf.trainable_variables())
    
    def _define_inputs(self):
        with self.graph.as_default():
            shape = [self.batch_size]
            shape.extend(self.data_shape)
            self.images = tf.placeholder(
                tf.float32,
                shape=shape,
                name='input_images'
            )
            self.labels = tf.placeholder(
                tf.int32,
                shape=[self.batch_size, self.n_classes],
                name='labels'
            )
            self.learning_rate = tf.placeholder(
                tf.float32,
                shape=[],
                name='learning_rate'
            )
            self.momentum = tf.placeholder(
                tf.float32,
                shape=[],
                name='momentum'
            )
            self.keep_prob = tf.placeholder(
                tf.float32,
                shape=[],
                name='keep_prob'
            )
            self.weight_decay = tf.placeholder(
                tf.float32,
                shape=[],
                name='weight_decay'
            )
            self.state_switch = tf.placeholder(
                tf.float32,
                shape=[None],
                name='state_switch'
            )
            self.is_training = tf.placeholder(tf.bool, shape=[])

    def _build_graph(self):
        with self.graph.as_default():
            l = self.images
            cnt = 1
            last = 0
            for x_size in self.layer_infor:
                mask = self.state_switch[last:last+x_size]
                last += x_size
                l = self.fullyconnected('fc{}'.format(cnt), l, x_size)
                l = tf.nn.relu(l)
                l = l*mask
                cnt += 1
            self.logits = self.fullyconnected('out', l, self.n_classes)
            self.prediction = tf.nn.softmax(self.logits)
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.labels
            ))
            self.cross_entropy = cross_entropy
            l2_loss = tf.add_n(
                [tf.nn.l2_loss(var) for (var) in tf.trainable_variables()]
            )

            optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
            self.train_step = optimizer.minimize(self.cross_entropy + l2_loss*self.weight_decay)

            self.correct_prediction = tf.equal(
                tf.argmax(self.prediction, 1),
                tf.argmax(self.labels, 1)
            )  
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.labels, 1), tf.argmax(self.prediction, 1), num_classes=self.n_classes)
    

    def fullyconnected(self, name, _input, out_channel):
        shape = _input.get_shape()
        if len(shape) == 4:
            size = shape[1].value * shape[2].value * shape[3].value
        else:
            size = shape[-1].value
        
        with tf.variable_scope(name):
            w = tf.get_variable(
                name='weights',
                shape=[size, out_channel],
                initializer=tf.contrib.layers.variance_scaling_initializer()
            )
            b = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[out_channel]))
            flat_x = tf.reshape(_input, [-1, size])
            output = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            print(output)
            return output

    def train_single_epoch(self, mnist, hyper_param):
        num_examples = mnist.train.num_examples
        total_acc = []
        total_los = []
        for i in range(1, (num_examples//self.batch_size) + 1):
            batch = mnist.train.next_batch(self.batch_size)
            images, labels = batch
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.momentum: hyper_param['momentum'],
                self.keep_prob: hyper_param['keep_prob'],
                self.learning_rate: hyper_param['learning_rate'],
                self.weight_decay: hyper_param['weight_decay'],
                self.state_switch: hyper_param['state_switch'],
                self.is_training: True
            }
            fetches = [self.train_step, self.cross_entropy, self.accuracy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy = result
            total_los.append(loss)
            total_acc.append(accuracy)

        mean_acc = np.mean(total_acc)
        mean_los = np.mean(total_los)

        return mean_acc, mean_los


    def test_single_epoch(self, mnist, hyper_param):
        total_acc = []
        num_test_examples = mnist.test.num_examples
        for _ in range(num_test_examples // self.batch_size):
            batch = mnist.test.next_batch(self.batch_size)
            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1],
                self.keep_prob: 1.0,
                self.weight_decay: hyper_param['weight_decay'],
                self.state_switch: hyper_param['state_switch'],
                self.is_training: False
            }
            fetches = [ self.accuracy ]
            accuracy = self.sess.run(fetches, feed_dict=feed_dict)
            total_acc.append(accuracy)
        mean_acc = np.mean(total_acc)
        return mean_acc

