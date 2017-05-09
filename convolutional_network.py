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
from sklearn.metrics import average_precision_score, precision_recall_curve
import sys
from os import path


train_poselet_id = None
n_poselets = None
attribute_id = None

if len(sys.argv) > 1:
    train_poselet_id = int(sys.argv[1])
if train_poselet_id < 0:
    n_poselets = -train_poselet_id
    train_poselet_id = None
if len(sys.argv) > 2:
    attribute_id = int(sys.argv[2])
    
PETA = True
if PETA:
    dataset_dir = '/files/data/PETA/'
else:
    dataset_dir = '/files/data/attributes_dataset/'
attr2idx, attr_list = get_attr_names(path.join(dataset_dir, "attributes.txt"))

ban_list = np.ones(len(attr2idx), dtype=bool)
for sel_atr in get_selected_attrs(path.join(dataset_dir, 'selected_attrs.txt')):
    ban_list[attr2idx[sel_atr]] = False
            
vgg_weights_filename = '/files/data/pretrain_model/vgg16_weights.npz'
precalc_fc_pattern = 'precalc_fc%d.dump'
# Parameters
if train_poselet_id is None:
    learning_rate = 1e-5 * 1
    weight_decay = 1e-2 * 6
else:
    learning_rate = 1e-5 * 5
    weight_decay = 1e-2 * 4
training_iters = 72000 #130000
batch_size = 16
display_step = 10
test_step = 900
# Network Parameters
#img_shape = 64, 64, 3
img_shape = 224, 224, 3
dropout = 0.50
pretrain_batch = 10


dev = 0.01
np.random.seed((hash(np) + hash(monotonic())) % 2**16)
    
    
def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# poselet_id==None means common pretrained layers
def convolitions(data_in, conv_layers, poselet_id):
    print("shape_in =", data_in.get_shape().as_list())
    d_in = data_in.get_shape().as_list()[3:]
    d_in = 1 if len(d_in) == 0 else d_in[0]
    for conv_block in conv_layers:
        for j, (i, r, c, d_out) in enumerate(conv_block):
            w = tf.get_variable("conv%d_%d_W_%s" % (i, j + 1, str(poselet_id)),
                                shape=[r, c, d_in, d_out],
                                initializer=tf.truncated_normal_initializer(stddev=dev),
                                trainable=poselet_id is not None)
            b = tf.get_variable("conv%d_%d_b_%s" % (i, j + 1, str(poselet_id)),
                                shape=[d_out],
                                initializer=tf.constant_initializer(dev),
                                trainable=poselet_id is not None)
            conv = tf.nn.conv2d(data_in, w, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.relu(tf.nn.bias_add(conv, b))
            d_in = d_out
            data_in = conv
        data_in = max_pool(data_in, 2)    
        print("mp shape =", data_in.get_shape().as_list())
    return data_in


def one_poselet(X, conv_layers, poselet_id):
    # Reshape input picture
    data_in = convolitions(X, conv_layers, poselet_id)
    # Fully connected layer
    fc_size = 1024
    data_size = np.prod(data_in.get_shape().as_list()[1:])
    w = tf.Variable(tf.truncated_normal([data_size, fc_size], stddev=dev),\
            name="fc1_W_%d" % poselet_id)
    b = tf.Variable(tf.constant(dev, shape=[fc_size]),\
            name="fc1_b_%d" % poselet_id)
    dense1 = tf.reshape(data_in, [-1, data_size])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, w), b))
    return dense1
    

def conv_net(X, conv_layers):
    fc_poselets = []
    for i in range(X.get_shape().as_list()[1]):
        fc_poselets.append(one_poselet(X[:,i,...], conv_layers, i))
    return fc_poselets


def fc_net(dense1, n_attr):
    res = []
    fc2_size = 256
    rng = range(n_attr)
    if attribute_id is not None:
        rng = [attribute_id]
    for i in rng:
        w = tf.Variable(tf.truncated_normal([dense1.get_shape().as_list()[1], fc2_size], \
            stddev=dev), name="fc2_W_%d" % i)
        b = tf.Variable(tf.constant(dev, shape=[fc2_size]), name="fc2_b_%d" % i)
        dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, w), b))
        dense2 = tf.nn.dropout(dense2, keep_prob)
        w = tf.Variable(tf.truncated_normal([fc2_size, 1], stddev=dev), 
                        name="fc3_W_%d" % i)
        b = tf.Variable(tf.constant(dev, shape=[1]), name="fc3_b_%d" % i)
        res.append(tf.add(tf.matmul(dense2, w), b)) #, dense1, dense2
    return tf.concat(axis=1, values=res)
        
    """
    outs = []
    for i in range(n_attr):
        w = tf.Variable(tf.truncated_normal([dense1.get_shape().as_list()[1], fc2_size], stddev=dev))
        b = tf.Variable(tf.constant(dev, shape=[fc2_size]))
        dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, w), b))
        dense2 = tf.nn.dropout(dense2, dropout)
        w = tf.Variable(tf.truncated_normal([fc2_size, 1], stddev=dev))
        b = tf.Variable(tf.constant(dev, shape=[1]))
        out = tf.add(tf.matmul(dense2, w), b)
        outs.append(out)
    return tf.reshape(tf.concat(axis=1, values=outs), [-1, n_attr])
    """

#64 -> 60 -> 30 -> 26 -> 13 -> 11 -> 6 -> 2 -> fc_576
pretrained_conv_layers = [
    [ [1, 3, 3, 64],
      [1, 3, 3, 64] ],
    #maxpooling
    [ [2, 3, 3, 128],
      [2, 3, 3, 128] ],
    #maxpooling
    [ [3, 3, 3, 256],
      [3, 3, 3, 256],
      [3, 3, 3, 256]]
    #maxpooling
]

conv_layers = [
    [ [4, 3, 3, 64] ], #512
    #maxpooling
    [ [5, 3, 3, 64] ], #512
    #maxpooling
]

#Load data
# Modes:
# 1. load ith poselet, then train
# 2. merge all poselets, and train full classifier
#Usage:
# convolitional_network.py [i]
x = None
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
data = None

if train_poselet_id is None:
    merged_poselets_dump = path.join(dataset_dir, 'merged_p.dump')
    if path.isfile(merged_poselets_dump):
        data = huge_load(merged_poselets_dump)
        print("merged poselets loaded")
    else:
        data = merge_poselets(n_poselets, path.join(dataset_dir, precalc_fc_pattern), ban_list=ban_list)
        huge_dump(data, merged_poselets_dump)
    px = tf.placeholder(tf.float32, [None, *data.train.px[0].shape])
    dense1 = px
else:
    data_dump = path.join(dataset_dir, 'data%d.dump' % train_poselet_id)
    if path.isfile(data_dump):
        data = huge_load(data_dump)
    else:
        data = AttributesDataset(dataset_dir, img_shape[:2], poselet_id=train_poselet_id,  ban_list=ban_list)
        huge_dump(data, data_dump)
    x = tf.placeholder(tf.float32, [None, *data.train.x[0].shape])
    print("x.shape =", data.train.x.shape)

    print("Dataset:")
    print("%d train examples, %d test" % (len(data.train.y), len(data.test.y)))  
    # tf Graph input
    rgb_subt = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, \
                            shape=[1, 1, 1, 3], name='rgb_subt')
            

    vggw = np.load(vgg_weights_filename)
    print("vgg weights loaded")

    X = tf.subtract(x, rgb_subt)
    x_poselets = []
    assigned = []
    with tf.variable_scope("preconv", reuse=None):
        for i in range(X.get_shape().as_list()[1]):
            x_poselets.append(tf.expand_dims( \
                convolitions(X[:,i,...], pretrained_conv_layers, None), 1))
    with tf.variable_scope("preconv", reuse=True):
        for conv_block in pretrained_conv_layers:
            for j, (i, r, c, d_out) in enumerate(conv_block):
                name = "conv%d_%d" % (i, j + 1)
                asg_b = tf.get_variable(name + '_b_%s' % str(None)).assign(vggw[name + '_b'])
                asg_W = tf.get_variable(name + '_W_%s' % str(None)).assign(vggw[name + '_W'])
                assigned += [asg_b, asg_W]

    pre_conv = tf.concat(axis=1, values=x_poselets)
    print("pre_conv shape: ", pre_conv.get_shape().as_list())
    px = tf.placeholder(tf.float32, pre_conv.get_shape().as_list())
    fc_poselets = conv_net(px, conv_layers)
    dense1 = tf.concat(axis=1, values=fc_poselets)

print("n_attr = %d" % data.n_attr)
print("attribute_id =", attribute_id)
y = tf.placeholder(tf.float32, [None, data.n_attr if attribute_id is None else 1])
data.attr_ids = np.arange(len(data.ban_attrs))[-data.ban_attrs]

pred0 = fc_net(dense1, data.n_attr)
print("out shape:", pred0.get_shape().as_list())
# Define loss and optimizer
if attribute_id is None:
    cost = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=pred0, labels=y),\
            tf.cast(tf.not_equal(y, 0.5), tf.float32)))
else:
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                          logits=pred0, labels=y))

# cost += wd * l2)loss
pred = tf.nn.sigmoid(pred0)
from scipy.optimize import minimize_scalar

learning_rate_l = 0.0001
learning_rate_r = 0.1

wd_l = 1e-3
wd_r = 1
steps = 100


print("trainable_variables:")
trainable = tf.trainable_variables()
sum_mem = 0
for v in trainable:
    print(v.name, v.get_shape().as_list())
    sum_mem += np.prod(v.get_shape().as_list())
print("sum mem:", sum_mem)

def apply_on_all_data(sess, graph, ph, data, batch_size=10):
    res = np.zeros((len(data),) + tuple(graph.get_shape().as_list()[1:]), dtype=np.float32)
    for i in range(0, len(data), batch_size):
        n = min(batch_size, len(data) - i)
        res[i:i+n] = sess.run(graph, feed_dict={ ph: data[i:i+n], keep_prob: 1. })
        print('.' if i * 70 // len(data) > \
                    (i - batch_size) * 70 // len(data) else '', end='')
        sys.stdout.flush()
    print('\n')
    return res
    

def merge_precision_recall(p, r, t):
    huge_dump((p, r, t), "merge_curves.dump")
    thrs = set()
    for ti in t:
        thrs = thrs.union(set(ti))
    thrs = np.array(sorted(list(thrs)))
    precision = np.zeros(len(thrs))
    recall = np.zeros(len(thrs))
    for i in range(len(thrs)):
        for j in range(len(t)):
            k = np.searchsorted(t[j], thrs[i])
            if k >= len(t[j]) or t[j][k] != thrs[i]:
                k -= 1
            if k >= 0:
                precision[i] += p[j][k]
                recall[i] += r[j][k]
        precision[i] /= len(t)
        recall[i] /= len(t)
    return precision, recall, thrs

    
def test_and_calc_mAP(sess, data_x, labels, attr_ids, name="res", attribute_id=None):
    print("Testing..", end='')
    res_pred = apply_on_all_data(sess, pred, px, data_x)
    res_pred.dump("%s_pred.dump" % name)
    if name is not None and path.isfile(path.join(dataset_dir, "%s_mAP.npz" % name)):
        mAP_npz = dict(np.load(path.join(dataset_dir, "%s_mAP.npz" % name)))
    else:
        mAP_npz = dict()
    mAP = 0.0
    n_attr = labels.shape[1]
    pp, tt, rr = [], [], []
    rng = range(n_attr)
    if attribute_id is not None:
        rng = [attribute_id]
    for i in rng:
        msk = labels[:, i] != 0.5
        p_i = average_precision_score(labels[msk, i], res_pred[msk, i * int(attribute_id is None)])
        precision, recall, thresholds = precision_recall_curve(labels[msk, i], 
                                                res_pred[msk, i * int(attribute_id is None)])
        pp.append(precision)
        rr.append(recall)
        tt.append(thresholds)
        plt.plot(recall, precision, label="%d" % (i + 1))
        acc_i = np.mean(labels[msk, i] == (res_pred[msk, i * int(attribute_id is None)] > 0.5).astype(np.float32))
        print("%s: AP = %.2f%%, acc=%.2f%%" % (attr_list[attr_ids[i]], p_i * 100, acc_i * 100), end=', ')
        prec = np.mean(labels[msk & (res_pred[:, i * int(attribute_id is None)] > 0.5), i])
        rec = np.mean((res_pred[labels[:, i] == 1.0, i * int(attribute_id is None)] > 0.5).astype(np.float32))
        f1 = 2 * prec * rec / (prec + rec)
        print("precision = %.2f%%, recall  = %.2f%%, F1 = %.2f%%" % (prec * 100, rec * 100, f1 * 100))
        print("attr_ids[i]", attr_ids[i])
        mAP_npz[attr_list[attr_ids[i]]] = np.array([p_i, acc_i, prec, rec, f1])
        mAP += p_i
    if attribute_id is None:
        mAP /= n_attr
    if name is not None:
        np.savez(path.join(dataset_dir, "%s_mAP.npz" % name), **mAP_npz)
    #precision, recall, thresholds = merge_precision_recall(pp, rr, tt)
    #plt.plot(recall, precision, "--", label="merged", c="black")
    plt.ylim(ymin=0, ymax=1)
    plt.xlim(xmin=0, xmax=1)
    plt.legend(loc='lower left', fontsize=10)
    plt.savefig('%s_precision-recall.png' % name)
    plt.clf()

    return mAP


def test(step_i):  
    m = np.power(wd_r / wd_l, 1.0 / steps)
    val = wd_l * np.power(m, step_i)
    wd = weight_decay
    #if learning_rate is not None:
    lr = learning_rate
    print("lr =", lr, "wd =", wd)
    # L2 loss
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost +  wd * l2_loss, var_list=trainable)

    #Softmax 
    print("Init all")
    # Initializing the variables
    res_acc = 0
    # Launch the graph
    plt_y = []
    plt_acc = []
    plt_acc_tr = []
    plt_it1 = []
    plt_it2 = []
    with tf.Session() as sess:
        print("Glob init.")
        sess.run(tf.global_variables_initializer())
        print("Assign pretrained.")
        print("Session started.")
        step = 1
        if train_poselet_id is None:
            if attribute_id is None:
                bg = BatchGenerator(data.train.px, data.train.y)
            else:
                bg = UniformBatchGen(data.train.px, data.train.y, attribute_id)
        else:
            for asg in assigned:
                sess.run(asg)
            pretrain_dump = path.join(dataset_dir, 'precalc_train%d.dump' % train_poselet_id)
            pretest_dump = path.join(dataset_dir, 'precalc_test%d.dump' % train_poselet_id)
            
            #if path.isfile(pretrain_dump):
                #precalc_train = huge_load(pretrain_dump)
            #else:
                #precalc_train = apply_on_all_data(sess, pre_conv, x, 
                                                #data.train.x, batch_size=pretrain_batch)
                #huge_dump(precalc_train, pretrain_dump)
            #batch_show(data.train.x[:16])
            
            precalc_train = apply_on_all_data(sess, pre_conv, x, 
                                            data.train.x, batch_size=pretrain_batch)
            del data.train.x
            data.train.px = precalc_train
            #if path.isfile(pretest_dump):
                #precalc_test = huge_load(pretest_dump)
            #else:
                #precalc_test = apply_on_all_data(sess, pre_conv, x, 
                                                #data.test.x, batch_size=pretrain_batch)
                #huge_dump(precalc_test, pretest_dump)
            precalc_test = apply_on_all_data(sess, pre_conv, x, 
                                            data.test.x, batch_size=pretrain_batch)
            del data.test.x
            data.test.px = precalc_test
        
            bg = BatchGenerator(precalc_train, data.train.y)
        #for i in range(x.get_shape().as_list()[1]):
        #    print("pre_conv shape: ", pre_conv.get_shape().as_list())
        
        tk = monotonic()
        try:
            while step * batch_size < training_iters:
                batch_xs, batch_ys = bg.next_batch(batch_size)
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
                sess.run(optimizer, feed_dict={px: batch_xs, y: batch_ys,
                                               keep_prob: dropout})
                if step % display_step == 0:
                    loss = sess.run(cost, feed_dict={px: batch_xs, y: batch_ys, keep_prob: 1.})
                    print("[%.1fs] Iter " % (monotonic() - tk) + str(step*batch_size) + \
                        ", Minibatch Loss= " + "{:.6f}".format(loss))
                    """
                    res_pr = sess.run(pred, feed_dict={ \
                                      x: batch_xs, keep_prob: 1. \
                                      })
                    print("batch:\n", res_pr)
                    """
                    #test_and_calc_mAP(sess, batch_xs, batch_ys, data.attr_ids, name=None, attribute_id=0)
                    plt_y.append(loss)
                    plt_it1.append(step*batch_size)
                    
                if step % test_step == 0:
                    mAP = test_and_calc_mAP(sess, data.test.px, data.test.y, 
                                            data.attr_ids, name="test", attribute_id=attribute_id)
                    print("Test mAP:", mAP)
                    mAP_tr = test_and_calc_mAP(sess, data.train.px, data.train.y, 
                                               data.attr_ids, name="train", attribute_id=attribute_id)
                    print("Train mAP:", mAP_tr)
                    plt_acc.append(mAP)
                    plt_acc_tr.append(mAP_tr)
                    plt_it2.append(step*batch_size)
                step += 1
        except KeyboardInterrupt:
            print("Interrupted")
        
        if False and train_poselet_id is None:
            plt.plot(plt_it1, plt_y)
            plt.show()
            plt.plot(plt_it2, np.transpose([plt_acc, plt_acc_tr]))
            plt.ylim(ymin=0.5, ymax=1)
            plt.show()
        
        print("Optimization Finished!")
        tmp = []
        for v in tf.global_variables():
            tmp.append(v.value().eval())
        huge_dump(tmp, "tmp.dump")
        mAP = test_and_calc_mAP(sess, data.test.px, data.test.y, data.attr_ids, 
                                name="test_full", attribute_id=attribute_id)
        print("Test mAP:", mAP)
        
        
        if train_poselet_id is not None:
            prefc_dump = path.join(dataset_dir, precalc_fc_pattern % train_poselet_id)
            precalc_train_fc = apply_on_all_data(sess, fc_poselets[0], px, 
                                            precalc_train, batch_size=pretrain_batch)
            data.train.px = precalc_train_fc
            del precalc_train
            precalc_test_fc = apply_on_all_data(sess, fc_poselets[0], px, 
                                            precalc_test, batch_size=pretrain_batch)
            data.test.px = precalc_test_fc
            huge_dump(data, prefc_dump)
        
            print("%dth poselet outputs dumped to: %s" % (train_poselet_id, prefc_dump))
    return -mAP

if False:
    opt_res = minimize_scalar(test, method='Bounded', 
                                bounds=[0, steps], 
                                options={'xatol':0.3})
    print("result:", opt_res.x)
else:
    test(0)
# learning_rate = 1e-5 * 5
# weight_decay = 1e-2 * 4
# 
#1th attr: 92.55%
#2th attr: 85.67%
#3th attr: 87.43%
#4th attr: 93.20%
#5th attr: 70.13%
#6th attr: 96.52%
#7th attr: 93.85%
#8th attr: 91.18%
#9th attr: 98.77%
#Test mAP: 89.92%

#64x64
#1th attr: 90.49%
#2th attr: 79.33%
#3th attr: 69.57%
#4th attr: 90.20%
#5th attr: 63.30%
#6th attr: 95.09%
#7th attr: 89.78%
#8th attr: 91.64%
#9th attr: 98.58%
#Test mAP: 0.853309656877



