import matplotlib.pyplot as plt
import numpy as np


def plot_init(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(0, 25):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i].reshape((28, 28)), cmap=plt.cm.binary)
        plt.xlabel(labels[i], fontsize=24)
    plt.show()


def plot_result(predictions, test_images, test_labels):
    rows = 5
    cols = 3
    plt.figure(figsize=(2 * 2 * cols, 2 * rows))
    for i in range(rows * cols):
        plt.subplot(rows, 2 * cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)

        plt.subplot(rows, 2 * cols, 2 * i + 2)
        plot_value(i, predictions, test_labels)
    plt.show()


def plot_image(index, predictions, test_labels, test_images):
    prediction, true_label, img = predictions[index], test_labels[index], test_images[index]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.binary)

    predicted_label = np.argmax(prediction)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * np.max(prediction),
                                         true_label),
               color=color, fontsize=24)


def plot_value(index, predictions, test_labels):
    prediction, true_label = predictions[index], test_labels[index]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    this_plot = plt.bar(range(10), prediction, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')
