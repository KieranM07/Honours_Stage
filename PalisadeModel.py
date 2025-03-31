import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import rasterio
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tkinter as tk
from tkinter import filedialog


main_dir = "PP_DATA2"
categories = ["Before_Fire", "Fire", "After_Fire", "No_Fire"]
X, y = [], []


def generate_global_max():

    # compute global min/max for thermal bands 10 and 11
    global_min10 = float("inf")
    global_max10 = float("-inf")
    global_min11 = float("inf")
    global_max11 = float("-inf")

    for category in categories:
        folder_path = os.path.join(main_dir, category)
        for file in os.listdir(folder_path):
            if file.endswith(".tif"):
                file_path = os.path.join(folder_path, file)

                with rasterio.open(file_path) as dataset:
                    thermal_band10 = dataset.read(10).astype(float)  # Read Band 10 (thermal)

                    thermal_band11 = dataset.read(11).astype(float)  # Read Band 11 (thermal)

                    global_min10 = min(global_min10, thermal_band10.min())
                    global_max10 = max(global_max10, thermal_band10.max())

                    global_min11 = min(global_min11, thermal_band11.min())
                    global_max11 = max(global_max11, thermal_band11.max())
                    
    return global_min10, global_max10, global_min11, global_max11


def load_data(global_min10, global_max10, global_min11, global_max11):
    # load, normalise and calc NDVI
    for category in categories:
        folder_path = os.path.join(main_dir, category)
        for file in os.listdir(folder_path):
            if file.endswith(".tif"):
                file_path = os.path.join(folder_path, file)

                with rasterio.open(file_path) as dataset:
                    # Read bands
                    thermal_band10 = dataset.read(10).astype(float)  # Read Band 10 (thermal)
                    thermal_band11 = dataset.read(11).astype(float)  # Read Band 11 (thermal)
                    nir_band = dataset.read(5).astype(float)      # NIR (Band 5)
                    red_band = dataset.read(4).astype(float)      # Red (Band 4)

                    # Global normalization for thermal bands
                    thermal_band10 = (thermal_band10 - global_min10) / (global_max10 - global_min10)
                    thermal_band11 = (thermal_band11 - global_min11) / (global_max11 - global_min11)

                    # NDVI calculation
                    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)  # Avoid divide by zero errors

                    # Stack bands
                    combined_bands = np.stack([thermal_band10, thermal_band11, ndvi], axis=-1)

                    X.append(combined_bands)
                    y.append(categories.index(category))


def architecture():

    # Define CNN model
    fire_model = models.Sequential([
        layers.Input(shape=(84, 84, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])

    fire_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return fire_model


def train(X, y, prediction_model):
    # Convert to a numpy array
    X = np.array(X)
    y = np.array(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Reshape for CNN (Add channel dimension)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Train the model
    prediction_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return X_test, y_test, prediction_model





def user_input():
    root = tk.Tk()
    root.withdraw()

    # Open file dialog
    user_file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.tif")])

    if user_file_path:
        print("file received")
    else:
        print("No file selected.")

    return user_file_path


def process_input(file_path, global_min10, global_max10, global_min11, global_max11):
    display_input(file_path)
    with rasterio.open(file_path) as dataset:
        # Read required bands
        thermal_band10 = dataset.read(10).astype(float)  # Thermal Band 10
        thermal_band11 = dataset.read(11).astype(float)  # Thermal Band 11
        nir_band = dataset.read(5).astype(float)  # NIR (Band 5)
        red_band = dataset.read(4).astype(float)  # Red (Band 4)

        # Global normalization for thermal bands
        thermal_band10 = (thermal_band10 - global_min10) / (global_max10 - global_min10)
        thermal_band11 = (thermal_band11 - global_min11) / (global_max11 - global_min11)

        # NDVI calculation
        ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)  # Avoid divide by zero errors

        # Stack bands
        combined_bands = np.stack([thermal_band10, thermal_band11, ndvi], axis=-1)
    return combined_bands


def display_input(fpath):
    with rasterio.open(fpath) as dataset:
        red_band = dataset.read(4).astype(float)
        blue_band = dataset.read(2).astype(float)
        green_band = dataset.read(3).astype(float)

        # Normalize each band to the range 0-255 for displaying
        red_band = ((red_band - red_band.min()) / (red_band.max() - red_band.min())) * 255
        blue_band = ((blue_band - blue_band.min()) / (blue_band.max() - blue_band.min())) * 255
        green_band = (((green_band - green_band.min()) / (green_band.max() - green_band.min())) * 255)
        rgb = np.dstack((red_band, green_band, blue_band)).astype(np.uint8)
        plt.imshow(rgb)
        plt.show()
def predict_input(bands, model):

    # Reshape input to match training shape
    input_data = np.expand_dims(bands, axis=-1)  # Ensure proper channel dimension
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    input_prediction = model.predict(input_data)

    predicted_label = np.argmax(input_prediction)

    return predicted_label


def train_model():
    min10, max10, min11, max11 = generate_global_max()
    load_data(min10, max10, min11, max11)
    model_arch = architecture()
    Test_X, Test_Y, fire_prediction_model = train(X, y, model_arch)
    return min10, max10, min11, max11, fire_prediction_model

def make_prediction(min_ten,max_ten,min_eleven,max_eleven,fire_model):
    inputted_file = user_input()
    user_bands = process_input(inputted_file, min_ten,max_ten,min_eleven,max_eleven)
    prediction = predict_input(user_bands, fire_model)

    result = categories[prediction]
    return result


#min10, max10, min11, max11, fire_prediction_model = train_model()
#make_prediction(min10, max10, min11, max11, fire_prediction_model)
