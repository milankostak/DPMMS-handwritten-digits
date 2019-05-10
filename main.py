import tensorflow as tf
import mnist
# import time
import ploting

# load train data set
train_images = mnist.train_images()
train_labels = mnist.train_labels()

# load test data set
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# reshape data for working with tensor flow Flatten layer
# then cast it to float so it can be normalized
train_images = train_images.reshape(60000, 28, 28, 1).astype('float32')
test_images = test_images.reshape(10000, 28, 28, 1).astype('float32')

# normalize data - pixel values form <0;255> to <0;1>
train_images /= 255
test_images /= 255

# show graph and check if train data labeling is correct
ploting.plot_init(train_images, train_labels)

# basic neural network with 3 hidden layers
# best accuracy: 0.9843
#
# activation - sigmoid, relu, relu6, leaky_relu, selu, tanh
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
#     tf.keras.layers.Dense(4096, activation=tf.nn.relu),
#     tf.keras.layers.Dense(1024, activation=tf.nn.relu),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# convolutional neural network
# best accuracy: 0.9933
#
# pooling - MaxPooling2D, AveragePooling2D
# padding - same, valid
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),
    tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(4096, activation=tf.nn.relu),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile neural network with default values for optimizer and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# determines how often to perform backpropagation
# bigger value can significantly speed up training
BATCH_SIZE = 1000

# train the neural network for 5 epochs - run through train data 5 times
# start = time.time()
model.fit(x=train_images, y=train_labels, epochs=5, batch_size=BATCH_SIZE)
# end = time.time()
# print("Time: " + str(end - start) + " s")

# evaluate neural network on test data set
test_loss, test_accuracy = model.evaluate(x=test_images, y=test_labels, steps=1)

print('\nAccuracy on test dataset: ', test_accuracy)

# show a fancy graph
predictions = model.predict(test_images)
ploting.plot_result(predictions, test_images, test_labels)

# way to open the file in python console
# exec(open("main.py").read())
