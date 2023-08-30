import os
from tensorflow.keras.utils import plot_model


def plot_model_architecture(model, output_dir='./logs'):
    os.makedirs(output_dir, exist_ok=True)
    plot_model(model, to_file=os.path.join(output_dir, 'model.png'), show_shapes=True, show_layer_names=True)
    print(f"Model architecture saved as PNG in {os.path.join(output_dir, 'model.png')}")
