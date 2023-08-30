import matplotlib.pyplot as plt
import numpy as np
import random


def visualize_plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


def visualize_predictions(model, generator, classes, num_images=12):
    # get the total number of batches in the generator
    num_batches = len(generator)

    # pick a random batch index
    batch_index = random.randint(0, num_batches - 1)

    # skip to the randomly chosen batch
    for _ in range(batch_index):
        _ = generator.next()

    # retrieve a batch of images from the generator
    val_images, val_labels = generator.next()

    plt.figure(figsize=(15, 15))

    # we don't visualize more images than we have available
    num_images = min(num_images, len(val_images))

    for i in range(num_images):
        plt.subplot(4, 4, i + 1)

        # predict the class of the image
        predictions = model.predict(np.expand_dims(val_images[i], axis=0), verbose=0)[0]
        true_label = classes[val_labels[i].argmax()]
        predicted_label = classes[predictions.argmax()]

        # if the prediction is correct, display the title in green, otherwise display it in red
        color = 'green' if true_label == predicted_label else 'red'

        plt.title(f"Predicted: {predicted_label}, \nTrue : {true_label} ", color=color, fontsize=10)
        plt.imshow(val_images[i])
        plt.axis('off')
    plt.subplots_adjust(top=0.85, wspace=0.5, hspace=1.5)
    plt.show()
