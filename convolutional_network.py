#!/usr/bin/env python3

'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import numpy as np
import joblib
from load_data import *
from time import monotonic
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score  
import sys

dataset_dir = '/files/data/attributes_dataset/'
# Parameters
learning_rate = 1e-4 * 5
weight_decay = 0.0005
training_iters = 400000
batch_size = 16
display_step = 10
test_step = 300
# Network Parameters
img_shape = 64, 64, 3
n_attr = 9
dropout = 0.75

dev = 0.05
np.random.seed((hash(np) + hash(monotonic())) % 2**16)
    
    

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def one_poselet(X, conv_layers, dropout):
    # Reshape input picture
    d_in = img_shape[2] if len(img_shape) > 2 else 1
    data_in = X
    #print("shape_in =", data_in.get_shape().as_list())
    for r, c, d_out in conv_layers:
        w = tf.Variable(tf.truncated_normal([r, c, d_in, d_out], stddev=dev))
        b = tf.Variable(tf.constant(dev, shape=[d_out]))
        conv = tf.nn.conv2d(data_in, w, strides=[1, 1, 1, 1], padding='VALID')
        conv = tf.nn.relu(tf.nn.bias_add(conv, b))
        conv = max_pool(conv, 2)
        conv = tf.nn.dropout(conv, dropout)
        d_in = d_out
        data_in = conv
        #print("shape =", data_in.get_shape().as_list())

    # Fully connected layer
    fc_size = 576
    data_size = np.prod(data_in.get_shape().as_list()[1:])
    w = tf.Variable(tf.truncated_normal([data_size, fc_size], stddev=dev))
    b = tf.Variable(tf.constant(dev, shape=[fc_size]))
    dense1 = tf.reshape(data_in, [-1, data_size])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, w), b))
    dense1 = tf.nn.dropout(dense1, dropout)
    return dense1
    
def conv_net(X, conv_layers, dropout):
    fc_poselets = []
    for i in range(X.get_shape().as_list()[1]):
        fc_poselets.append(one_poselet(X[:,i,...], conv_layers, dropout))
    dense1 = tf.concat(axis=1, values=fc_poselets)
    
    fc2_size = 128
    w = tf.Variable(tf.truncated_normal([dense1.get_shape().as_list()[1], fc2_size], stddev=dev))
    b = tf.Variable(tf.constant(dev, shape=[fc2_size]))
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, w), b))
    dense2 = tf.nn.dropout(dense2, dropout)
    w = tf.Variable(tf.truncated_normal([fc2_size, n_attr], stddev=dev))
    b = tf.Variable(tf.constant(dev, shape=[n_attr]))
    return tf.add(tf.matmul(dense2, w), b), dense1, dense2

#64 -> 60 -> 30 -> 26 -> 13 -> 11 -> 6 -> 2 -> fc_576
conv_layers = [
    [5, 5, 64],
    [5, 5, 64],
    [3, 3, 64],
    [3, 3, 64]
]

#Load data
try:
    data = joblib.load('data.dump')
except:
    data = AttributesDataset(dataset_dir, img_shape[:2], n_attr)
    joblib.dump(data, 'data.dump', compress=9)
print("Dataset:")
print("%d train examples, %d test" % (len(data.train.y), len(data.test.y)))  
# tf Graph input
x = tf.placeholder(tf.float32, [None, *data.train.x[0].shape])
y = tf.placeholder(tf.float32, [None, n_attr])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Construct model
pred0, d1, d2 = conv_net(x, conv_layers, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred0, labels=y), 
                                  tf.cast(tf.not_equal(y, 0.5), tf.float32)))
# L2 loss
print("trainable_variables:", len(tf.trainable_variables()))
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
#cost += weight_decay * l2_loss

pred = tf.nn.sigmoid(pred0)
from scipy.optimize import minimize_scalar

learning_rate_l = 0.0001
learning_rate_r = 0.1
steps = 100

def test_and_calc_mAP(sess, xy):
    test_batch_size = 100
    print("Testing..", end='')
    res_pred = np.zeros((0, n_attr), dtype=np.float32)
    for i in range(0, len(xy.x), test_batch_size):
        n = min(test_batch_size, len(xy.x) - i)
        res_pred = np.concatenate((res_pred, sess.run(pred, feed_dict={ \
                                    x: xy.x[i:i+n], keep_prob: 1. \
                                    })))
        print('.' if i * 70 // len(xy.x) > \
                    (i - test_batch_size) * 70 // len(xy.x) else '', end='')
        sys.stdout.flush()
    print('\n')
    res_pred.dump("res_pred.dump")
    mAP = 0.0
    labels = xy.y
    for i in range(n_attr):
        msk = labels[:, i] != 0.5
        p_i = average_precision_score(labels[msk, i], res_pred[msk, i])
        print("%dth attr: %.2f%%" % (i+1, p_i * 100))
        mAP += p_i
    mAP /= n_attr
    return mAP


def test(step_i, learning_rate=None):  
    m = np.power(learning_rate_r / learning_rate_l, 1.0 / steps)
    lr = learning_rate_l * np.power(m, step_i)
    if learning_rate is not None:
        lr = learning_rate
    print("lr =", lr)
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    #Softmax 

    print("Init all")
    # Initializing the variables
    init = tf.global_variables_initializer()

    res_acc = 0
    # Launch the graph
    plt_y = []
    plt_acc = []
    plt_acc_tr = []
    plt_it1 = []
    plt_it2 = []
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        #p = sess.run(pred0, feed_dict={x: data.test.x[:100], y: data.test.y[:100, :n_attr], keep_prob: 1.}).ravel()
        #print(data.test.y[:100, :n_attr].ravel(), '\n', p)
        #print(sess.run(pred, feed_dict={x: data.test.x[:100], y: data.test.y[:100, :n_attr], keep_prob: 1.}).ravel())
        
        
        tk = monotonic()
        try:
            while step * batch_size < training_iters:
                batch_xs, batch_ys = data.next_batch(batch_size)
                #for i in range(len(batch_xs)):
                #    print(batch_ys[i])
                #batch_show(batch_xs)
                
                """
                summ = 0.0
                for v in tf.global_variables():
                    summ += np.sum(v.value().eval())
                print(summ)
                """
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                               keep_prob: dropout})
                if step % display_step == 0:
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    print("[%.1fs] Iter " % (monotonic() - tk) + str(step*batch_size) + \
                        ", Minibatch Loss= " + "{:.6f}".format(loss))
                    """
                    res_pr = sess.run(pred, feed_dict={ \
                                      x: batch_xs, keep_prob: 1. \
                                      })
                    print("batch:\n", res_pr)
                    """
                    plt_y.append(loss)
                    plt_it1.append(step*batch_size)
                    
                if step % test_step == 0:
                    mAP = test_and_calc_mAP(sess, data.test)
                    print("Test mAP:", mAP)
                    mAP_tr = test_and_calc_mAP(sess, data.train)
                    print("Train mAP:", mAP_tr)
                    plt_acc.append(mAP)
                    plt_acc_tr.append(mAP_tr)
                    plt_it2.append(step*batch_size)
                step += 1
        except KeyboardInterrupt:
            print("Interrupted")
        plt.plot(plt_it1, plt_y)
        plt.show()
        plt.plot(plt_it2, plt_acc)
        plt.show()
        plt.plot(plt_it2, plt_acc_tr)
        plt.show()
        print("Optimization Finished!")
        tmp = []
        for v in tf.global_variables():
            tmp.append(v.value().eval())
        joblib.dump(tmp, "tmp.dump", compress=9)
        mAP = test_and_calc_mAP(sess, data.test)
        """
        print('pred0\n',sess.run(pred0, feed_dict={x: data.test.x[0:4], keep_prob: 1. }))
        print('d1\n',sess.run(d1, feed_dict={x: data.test.x[0:4], keep_prob: 1. }))
        print('d2\n',sess.run(d2, feed_dict={x: data.test.x[0:4], keep_prob: 1. }))
        """
        #print(result)
        #print(data.test.y[:test_batch_size])
        
    return -mAP

if False:
    opt_res = minimize_scalar(test, method='Bounded', 
                                bounds=[0, steps], 
                                options={'xatol':0.3})
    print("result:", opt_res.x)
else:
    test(0, learning_rate=learning_rate)

#best 0.568889
#     0.584444
# 50.62 with0.5, skel
# 53.89 - with0.5, full roi 
