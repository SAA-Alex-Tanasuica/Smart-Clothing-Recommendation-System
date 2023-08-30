import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD


def create_resnet50_model(num_classes):
    print("Creating ResNet50 model...")

    # load the ResNet50 model with pretrained weights from imagenet and without the final classification layer
    print("Loading the ResNet50 base model with ImageNet weights...")
    model = ResNet50(input_shape=(224, 224, 3),
                     include_top=False,
                     weights='imagenet')

    # make all layers trainable
    print("Setting base model layers as trainable...")
    for layer in model.layers:
        layer.trainable = True

    # define new classification layer
    print("Adding custom classification layers...")
    starting_classification_layer = model.input

    # define ResNet50 base model
    x = model.output

    # add a GlobalAveragePooling2D layer - this helps reduce the dimensionality of the data
    # and can often lead to more efficient training than Flatten
    x = GlobalAveragePooling2D()(x)

    # add a Dropout layer
    x = Dropout(0.2)(x)

    # add a Dense layer with 512 neurons
    x = Dense(512, activation='relu')(x)

    # add another Dropout layer
    x = Dropout(0.2)(x)

    # add another Dense layer
    x = Dense(128, name='Feature_extractor', activation='relu')(x)

    # output
    custom_classification_layers = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # create a new model
    print("Creating the final model...")
    model = tf.keras.Model(starting_classification_layer, custom_classification_layers)

    print("Model creation complete!")

    return model


def compile_model(model):
    print("Compiling model...")
    model.compile(SGD(learning_rate=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model summary:")
    model.summary()
    print("Model compiled!")
    return model
