""""tensorflow实现进阶的卷积网络的例子，数据集cifar-10
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
"""

import tensorflow as tf
import numpy as np
import time
import math
from src import cifar10, cifar10_input

max_step = 3000#训练轮数
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
#初始化weight权重，添加L2正则化处理
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var
#下载cifar10数据集
cifar10.maybe_download_and_extract()
#利用cifar10_input中的函数产生训练需要使用的数据
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
#利用cifar10_input中的函数产生测试数据
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])
#卷积第一层
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)#LRN
#卷积第二层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value#获取数据扁平化之后的长度
weight3 = variable_with_weight_loss(shape=[dim, 384],stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3)+bias3)#全连接层和ReLU的激活函数
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4)+bias4)#全连接层和ReLU的激活函数
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5= tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

loss=loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()
#正式训练
for step in range(1000):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    loss_value = sess.run([train_op, loss], feed_dict={image_holder:image_batch, label_holder:label_batch})
    duration = time.time()-start_time
    if step % 10 == 0:
        examples_per_sec = float(batch_size/duration)
        sec_per_batch = float(duration)
        # format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        # print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
        print(step, loss_value, examples_per_sec, sec_per_batch)
#评测模型再测试集上的准确率
num_examples = 10000
num_iter = int(math.ceil(num_examples/batch_size))
true_count = 0
total_sample_count = num_iter*batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1
precision = true_count/total_sample_count
print('precision @ 1=%.3f' % precision)



