import gzip
import os

import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def get_data():
    base = './data'
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = []
    for fname in files:
        paths.append(os.path.join(base, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train), (x_test, y_test)


def get_data2():
    path = 'mnist.npz'
    origin_folder = './dataset/'
    path = get_file(
        path,
        origin=origin_folder + 'mnist.npz',
        file_hash='8a61469f7ea1b51cbae51d4f78837e45')

    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)


(train_images, train_labels), (test_images, test_labels) = get_data()
print(train_images.shape)
print(test_images.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
inputs = tf.keras.Input(shape=(1, 28, 28))  # Returns a placeholder tensor
# A layer instance is callable on a tensor, and returns a tensor.
# x = layers.Flatten(input_shape=(128, 128))(inputs)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
print(x)
x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_first')(x)
print(x)
x = layers.Dropout(0.5)(x)
print(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(x)
print(x)
x = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(x)
print(x)
x = layers.Dropout(0.35)(x)
print(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(x)
print(x)
x = layers.MaxPool2D(pool_size=(2, 2), padding='same', data_format='channels_first')(x)
print(x)
x = layers.Dropout(0.5)(x)
print(x)
x = layers.Flatten()(x)
print(x)
y = layers.Dense(625, activation='relu')(x)  # 全连接层
print(x)
y = layers.Dropout(0.5)(y)
print(x)
predictions = layers.Dense(10, activation='softmax')(y)
print(predictions)

model = tf.keras.Model(inputs=inputs, outputs=predictions)
train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
model.compile(optimizer=tf.train.AdadeltaOptimizer(0.005),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=tf.keras.metrics.categorical_accuracy)
tx = np.empty([60000, 28, 28, 1])
tx[:, :, :, 0] = train_images
tex = np.empty([10000, 28, 28, 1])
tex[:, :, :, 0] = test_images

model.fit(tx, train_labels, verbose=1, epochs=120, batch_size=128, validation_data=(tex, test_labels))

test_loss, test_acc = model.evaluate(tex, test_labels, verbose=0)

print('Test accuracy:', test_acc)

predictions = model.predict(tex)

print(np.argmax(predictions[0]))

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

json_string = model.to_json()
print(json_string)
model.save('my_model.h5')

