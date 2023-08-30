import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from temperature_mapping import temperature_classes


def forecast_images(model, classes, img_forecast_folder_path, temperature):
    custom_paths = [os.path.join(img_forecast_folder_path, img) for img in os.listdir(img_forecast_folder_path)]
    path = random.choice(custom_paths)
    image = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = input_arr.astype('float32') / 255.
    predictions = model.predict(input_arr, verbose=0)
    predicted_classes = np.argsort(predictions)
    predictions.sort()

    top_classes_temperatures = [
        (classes[predicted_classes[0][-1]], round(predictions[0][-1] * 100, 2),
         get_temperature_category([classes[predicted_classes[0][-1]]])),
        (classes[predicted_classes[0][-2]], round(predictions[0][-2] * 100, 2),
         get_temperature_category([classes[predicted_classes[0][-2]]])),
        (classes[predicted_classes[0][-3]], round(predictions[0][-3] * 100, 2),
         get_temperature_category([classes[predicted_classes[0][-3]]])),
    ]

    title = "\n".join([f"{cls}  -  {perc}%  -  {temp_range}°C" for cls, perc, temp_range in top_classes_temperatures])

    plt.title(title, color='black', fontsize=10)

    plt.imshow(image)
    plt.axis('off')
    plt.text(0.5, -0.1, f"Current temperature: {temperature}°C", ha='center', va='center',
             transform=plt.gca().transAxes)


def visualize_forecast_images(model, classes, img_forecast_folder_path, temperature, num_images=12):
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        forecast_images(model, classes, img_forecast_folder_path, temperature)
    plt.subplots_adjust(top=0.85, wspace=0.5, hspace=1.5)
    plt.show()


def get_temperature_category(predicted_classes):
    temperature_ranges = \
        [temperature_classes.get(predicted_class, "Range not known") for predicted_class in predicted_classes]
    return temperature_ranges
