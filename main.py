import tensorflow as tf
import math
import mnist
# import time
import ploting

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = train_images.reshape(60000, 28, 28, 1).astype('float32')
test_images = test_images.reshape(10000, 28, 28, 1).astype('float32')

# normalize data
train_images /= 255
test_images /= 255

num_train_examples = train_images.shape[0]
num_test_examples = test_images.shape[0]

# check if data labeling is correct
ploting.plot_init(train_images, train_labels)

# activation - sigmoid, relu, relu6, leaky_relu, selu, tanh
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
# pooling - MaxPooling2D, AveragePooling2D
# padding - same, valid
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE = 1000

# training
# start = time.time()
model.fit(x=train_images, y=train_labels, epochs=5, batch_size=BATCH_SIZE)
# model.fit(x=train_images, y=train_labels, epochs=5, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE))
# end = time.time()
# print("Time: " + str(end - start) + " s")

# evaluating
test_loss, test_accuracy = model.evaluate(x=test_images, y=test_labels, steps=math.ceil(num_test_examples / BATCH_SIZE))

print('\nAccuracy on test dataset: ', test_accuracy)

predictions = model.predict(test_images)
ploting.plot_result(predictions, test_images, test_labels)
