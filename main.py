import numpy as np
from weather import get_weather
from load_data import load_dataset
from process_data import process_data, show_images
from data_generator import get_data_generators
from resnet50 import create_resnet50_model, compile_model
from model_plot import plot_model_architecture
from train_model import train_model
from tensorflow.keras.models import load_model
from tensorflow import keras
from visualize_data import visualize_plot_loss_curves, visualize_predictions
from forecast import visualize_forecast_images

if __name__ == '__main__':
    '''
    city_name = input("Introduce your city name: \n")
    '''
    city_name = 'Craiova'
    api_key = "2caa892484c6c501c0d3ae93dd6348b2"
    temperature = get_weather(city_name, api_key)
    print(temperature)

    csv_path = './data/fashion_dataset.csv'
    img_folder_path = './data/images'

    # batch loading
    batch_size = 3500
    num_batches = 1  # load 2500 images in total
    for i in range(num_batches):
        start_row = i * batch_size
        images, labels = load_dataset(csv_path, img_folder_path, nrows=batch_size)
        print(f"Loading done for {num_batches} batches of size {batch_size}.")

    print("Loading successful!")
    print(images.shape)
    print(labels.shape)

    start_row = 0
    X_train, y_train, df, class_dict = process_data(csv_path, img_folder_path, start_row, batch_size)

    show_images(df, img_folder_path)
    print("Data process successful!")

    generator, validate = get_data_generators(df, img_folder_path)

    num_classes = 17  # there are 17 classes within the images
    '''
    model = create_resnet50_model(num_classes)

    model = compile_model(model)

    plot_model_architecture(model)

    model, history = train_model(model, generator, validate)

    model.save('./models/SCRS_Trained_Model_ResNet50Epoch20.tf')
    '''
    model = keras.models.load_model('./models/SCRS_Trained_Model_ResNet50Epoch20.tf')
    print('Model loaded successfully!')

    print('Calculating accuracy...')
    loss, accuracy = model.evaluate(validate, verbose=2)
    print(f'Loss: {loss:.2f}')
    print(f'Accuracy: {100 * np.round(accuracy, 2)}%')
    '''
    # in case we want to visualize the plot loss 
    visualize_plot_loss_curves(history)  # however we only save the history during training of the model
    '''
    print('Visualizing the prediction:')
    keys = list(generator.class_indices.keys())
    visualize_predictions(model, validate, keys)
    print('Visualization finished!')

    img_forecast_folder_path = './data/forecast_images'
    print('Forecasting the image percentages:')
    visualize_forecast_images(model, keys, img_forecast_folder_path, temperature)

    print('Program completed!')
