import tensorflow as tf 
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
# read TFRecord data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
InitData()

SetParameters()


TrainModel()



i=0
while XXXXXX :
  print RecognCall(image[i])
  i++




def label RecognCall(image)




'''


def parse_image_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    return features


reader = tf.TFRecordReader()
dir_train = "data/fashion_train.tfrecords"
dir_test = "data/fashion_test.tfrecords"
test_queue = tf.train.string_input_producer([dir_test])
train_queue = tf.train.string_input_producer([dir_train])

_, serialized_example = reader.read(train_queue)
_, serialized_test = reader.read(test_queue)

features1 = parse_image_example(serialized_example)
features_test = tf.parse_single_example(
    serialized_test,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
    })
image = tf.decode_raw(features1['image_raw'], tf.uint8)
print(image)
image.set_shape([784])
print(image)
"""
print(image.op)
print(image.dtype)
print(image.value_index)
"""

label = tf.cast(features1['label'], tf.int32)
print(label)
image_test = tf.decode_raw(features_test['image_raw'], tf.uint8)
image_test.set_shape([784])
label_test = tf.cast(features_test['label'], tf.int32)
pixel = tf.cast(features1['pixels'], tf.int32)


batch_size = 128
test_size = 256
# generate train batch data

min_after_dequeue = 10000
capacity = min_after_dequeue + 3*batch_size

image_batch, lb = tf.train.batch(
   [image, label], batch_size=batch_size, capacity=capacity)
print(image_batch)
label_batch = tf.one_hot(lb, 10)
image_batch_test, lb_test = tf.train.batch(
  [image_test, label_test], batch_size=100, capacity=10000)
label_batch_test = tf.one_hot(lb_test, 10)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128*4*4, 625])

w_o = init_weights([625, 10])
ckpt_dir = "./ckpt_dir2"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver()
non_storable_variable = tf.Variable(777)


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # flatten
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))  # juzhenchen
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    pyx = tf.matmul(l4, w_o, name="pyx")
    return pyx


p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
print("py_x", py_x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1, "predict_op")
print(predict_op)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print(ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    start = global_step.eval()
    print("start from: ", start)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(start, 11):
        accuracy = 0
        for ii in range(100):
            bx, batch_ys = sess.run([image_batch, label_batch])
            batch_xs = bx.reshape(-1, 28, 28, 1)
            sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys, p_keep_conv: 0.8, p_keep_hidden: 0.5})
        for jj in range(100):
            teX, teY =sess.run([image_batch_test, label_batch_test])
            teX =teX.reshape(-1, 28, 28, 1)
            accuracy = accuracy+np.mean(np.argmax(teY, axis=1) == \
                                        sess.run(predict_op, feed_dict={X: teX, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
        print(i, accuracy/100)
        global_step.assign(i+1).eval()
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
    coord.request_stop()
    coord.join(threads)

    pic_dir = 'test_num'
    files = os.listdir(pic_dir)
    cnt = len(files)
    for i in range(cnt):
        files[i] = pic_dir+"/"+files[i]
        img = Image.open(files[i])
        print("input: ", files[i])
        imga = np.array(img)
        imgb = imga.reshape(-1, 28, 28, 1)
        print("output: ", predict_op.eval(feed_dict={X: imgb, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
        print("\n")


