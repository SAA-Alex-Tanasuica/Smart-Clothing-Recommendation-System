import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import UnidentifiedImageError
import numpy as np
import random


def process_data(csv_file, img_folder, start_row=0, batch_size=500):
    print("Starting data processing...")

    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['image'] = df['image'] + '.jpg'

    print("Removing corrupted images and unwanted labels...")
    corrupt_images = []
    df = df.drop(df[df['image'].isin(corrupt_images)].index, axis=0)
    df = df.drop(df[df['label'] == 'Not sure'].index)
    df = df.drop(df[df['label'] == 'Other'].index)

    print(f"Selecting batch of images from {start_row} to {start_row + batch_size}")
    df = df.iloc[start_row:start_row + batch_size]  # this is where we limit the data

    print("Mapping labels to integer values...")
    y_train = df['label']
    label_mapping = {label: index for index, label in enumerate(y_train.unique())}

    print("Loading and normalizing images...")

    def load_image(image):
        img = load_img(img_folder + '/' + image, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        return img_array

    x_train = np.stack(df['image'].apply(load_image))
    y_train = y_train.map(label_mapping).to_numpy()

    print("Data processing completed.")
    return x_train, y_train, df, label_mapping


def view_training_images(df, image_folder):
    index_randomized = random.choice(df.index)
    image_path = df['image'][index_randomized]
    full_path = image_folder + '/' + image_path
    img = mpimg.imread(full_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(df['label'][index_randomized])


def show_images(df, img_folder):
    plt.figure(figsize=(15, 15))
    for i in range(12):
        plt.subplot(4, 4, i+1)
        view_training_images(df, img_folder)
    plt.subplots_adjust(top=0.85, wspace=0.5, hspace=1.5)
    plt.show()
