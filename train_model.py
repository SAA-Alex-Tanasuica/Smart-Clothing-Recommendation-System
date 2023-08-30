import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def train_model(model, generator, validate, epochs=20):
    print("Training model...")

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                factor=0.5,
                                                min_lr=0.00001)

    history = model.fit(
        generator,
        epochs=epochs,
        validation_data=validate,
        callbacks=[es, learning_rate_reduction]
    )
    return model, history
