import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def get_data_generators(df, path, image_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=20,  # range to rotate pictures
                                 width_shift_range=0.1,  # width range to randomly translate pictures
                                 height_shift_range=0.1,  # height range to randomly translate pictures
                                 shear_range=0.2,  # shear angle counter-clockwise - causes the image to have a tint
                                 zoom_range=0.2,  # range within which to randomly zoom pictures
                                 horizontal_flip=True,  # randomly flipping half of the images horizontally
                                 fill_mode='nearest',  # points outside the boundaries of the input are filled
                                 validation_split=0.2  # fraction of images reserved for validation
                                 )

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=path,
        x_col='image',
        y_col='label',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset='training'
    )

    validate = datagen.flow_from_dataframe(
        dataframe=df,
        directory=path,
        x_col='image',
        y_col='label',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False,
        subset='validation'
    )

    print(f"Created train generator with {generator.n} samples and {np.unique(generator.class_indices)} classes.")
    print(f"Created validation generator with {validate.n} samples and {np.unique(validate.class_indices)} classes.")

    return generator, validate
