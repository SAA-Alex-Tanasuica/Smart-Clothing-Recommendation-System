import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_dataset(csv_path, img_folder_path, nrows):
    df = pd.read_csv(csv_path, nrows=nrows)
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = img_folder_path + '/' + row['image'] + '.jpg'  # add the image file extension
        print(f"Loading image: {img_path}")
        try:
            img = load_img(img_path, target_size=(224, 224))  # model requiring this shape
            img_array = img_to_array(img) / 255.0  # convert image to array and normalize to [0, 1]
            label = row['label']

            labels.append(label)
            images.append(img_array)
        except Exception as e:
            print(f"Failed to load image: {img_path}. Exception: {str(e)}")
            pass
    # convert lists to numpy arrays for machine learning model
    images = np.array(images)
    labels = np.array(labels)

    return images, labels
