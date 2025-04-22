import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import rasterio
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
import tkinter as tk
from tkinter import filedialog


main_dir = "Training_Data"
categories = ["High chance", "Live fire", "Low chance", "Little to no chance"]
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
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(192, activation='relu'),
        layers.Dense(4, activation='softmax')

    ])
    optimizer = optimizers.Adam(learning_rate=0.0025969815749328747)
    fire_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return fire_model


def train(X, y, prediction_model):
    # Convert to a numpy array
    X = np.array(X)
    y = np.array(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Reshape for CNN (Add channel dimension)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Train the model
    history = prediction_model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0, validation_data=(X_test, y_test))

    return X_test, y_test, prediction_model, history





def user_input():
    root = tk.Tk()
    root.withdraw()

    # Open file dialog
    user_file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.tif")])

    return user_file_path


def process_input(file_path, global_min10, global_max10, global_min11, global_max11):
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
        image = display_input(file_path)
    return combined_bands,image

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
        return rgb


def predict_input(bands, model):

    # Reshape input to match training shape
    input_data = np.expand_dims(bands, axis=-1)  # Ensure proper channel dimension
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    input_prediction = model.predict(input_data, verbose = 0)

    predicted_label = np.argmax(input_prediction)

    return predicted_label


def train_model():
    min10, max10, min11, max11 = generate_global_max()
    load_data(min10, max10, min11, max11)
    model_arch = architecture()
    Test_X, Test_Y, fire_prediction_model, hist = train(X, y, model_arch)
   # evaluation(Test_X, Test_Y, fire_prediction_model, hist)
    return min10, max10, min11, max11, fire_prediction_model


def make_prediction(min_ten,max_ten,min_eleven,max_eleven,fire_model):
    inputted_file = user_input()
    user_bands,img = process_input(inputted_file, min_ten,max_ten,min_eleven,max_eleven)
    prediction = predict_input(user_bands, fire_model)

    result = categories[prediction]
    return result,img

def recommend_actions(prediction):
    action = None
    if prediction == "Little to no chance":
        action = "No action required"
    elif prediction == "Low chance":
        action = "Continue to monitor area"
    elif prediction == "Live fire":
        action = "Evacuate area if in immediate danger and call the fire department"
    elif prediction == "High chance":
        action = "Inform fire department and put necessary precautions in place"
    else:
        action = "Error"

    return action

# def evaluation(test_X, test_Y, prediction_model, history):
#     y_pred = []
#     y_actual = []
#
#     # Iterate over the entire test dataset
#     for index in range(len(test_X)):
#         prediction = prediction_model.predict(test_X[index:index + 1])
#         predicted_label = np.argmax(prediction)  # Get index of highest probability
#         actual_label = test_Y[index]
#
#         y_pred.append(predicted_label)
#         y_actual.append(actual_label)
#
#     # Convert lists to arrays
#     y_pred = np.array(y_pred)
#     y_actual = np.array(y_actual)
#
#     # Create confusion matrix
#     conf_matrix = confusion_matrix(y_actual, y_pred)
#
#     plt.figure(figsize=(12, 8))
#
#     # Plot Prediction Heatmap
#     plt.subplot(2, 2, 1)
#     categories = ["High_Chance", "Live_Fire", "Low_Chance", "Fire_Unlikely"]
#     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
#     plt.xlabel("Predicted Label")
#     plt.ylabel("Actual Label")
#     plt.title("Prediction Heatmap")
#
#     # Plot Model Accuracy
#     plt.subplot(2, 2, 2)
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model Accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#
#     # Plot Model Loss
#     plt.subplot(2, 1, 2)
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#
#     plt.tight_layout()  # Prevent overlap
#     plt.show()

# def load_test_data(global_min10, global_max10, global_min11, global_max11):
#     # load, normalise and calc NDVI
#     X, y = [], []
#     test_dir = "Test data"
#     for folder in os.listdir(test_dir):
#         if folder == main_dir:
#             continue
#         folder_path = os.path.join(test_dir, folder)
#         if os.path.isdir(folder_path):
#             for category in categories:
#                 category_path = os.path.join(folder_path, category)
#                 if os.path.isdir(category_path):
#                     for file in os.listdir(category_path):
#                         if file.endswith(".tif"):
#                             file_path = os.path.join(category_path, file)
#
#                             with rasterio.open(file_path) as dataset:
#                                 # Read bands
#                                 thermal_band10 = dataset.read(10).astype(float)  # Read Band 10 (thermal)
#                                 thermal_band11 = dataset.read(11).astype(float)  # Read Band 11 (thermal)
#                                 nir_band = dataset.read(5).astype(float)      # NIR (Band 5)
#                                 red_band = dataset.read(4).astype(float)      # Red (Band 4)
#
#                                 # Global normalization for thermal bands
#                                 thermal_band10 = (thermal_band10 - global_min10) / (global_max10 - global_min10)
#                                 thermal_band11 = (thermal_band11 - global_min11) / (global_max11 - global_min11)
#
#                                 # NDVI calculation
#                                 ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)  # Avoid divide by zero errors
#
#                                 # Stack bands
#                                 combined_bands = np.stack([thermal_band10, thermal_band11, ndvi], axis=-1)
#
#                                 X.append(combined_bands)
#                                 y.append(categories.index(category))
#     return X, y

# def new_data(testX, testY, prediction_model):
#     y_pred = []
#     y_actual = []
#     print(len(testX))
#     # Iterate over the entire test dataset
#     for index in range(len(testX)):
#         testX[index] = np.expand_dims(testX[index], axis=0).astype(np.float32)
#         prediction = prediction_model.predict(testX[index:index + 1])
#         predicted_label = np.argmax(prediction)  # Get index of highest probability
#         actual_label = testY[index]
#
#         y_pred.append(predicted_label)
#         y_actual.append(actual_label)
#
#     # Convert lists to arrays
#     y_pred = np.array(y_pred)
#     _actual = np.array(y_actual)
#
#     conf_matrix = confusion_matrix(y_actual, y_pred)
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
#     plt.xlabel("Predicted Label")
#     plt.ylabel("Actual Label")
#     plt.title("Prediction Heatmap")
#     plt.show()
#
#
# min10, max10, min11, max11, fire_prediction_model = train_model()
# testx, testy = load_test_data(min10, max10, min11, max11)
# new_data(testx,testy,fire_prediction_model)
#make_prediction(min10, max10, min11, max11, fire_prediction_model)



# Hyperparameter tuning

# import keras_tuner as kt
# from tensorflow import keras
#
#
#
# def build_model(hp):
#     model = models.Sequential()
#     model.add(layers.Input(shape=(84, 84, 3)))
#
#     # Conv Layer 1
#     model.add(layers.Conv2D(
#         filters=hp.Choice('conv1_filters', [32, 64]),
#         kernel_size=hp.Choice('conv1_kernel', [3, 5]),
#         activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#
#     # Conv Layer 2
#     model.add(layers.Conv2D(
#         filters=hp.Choice('conv2_filters', [64, 128]),
#         kernel_size=hp.Choice('conv2_kernel', [3, 5]),
#         activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#
#     # Conv Layer 3
#     model.add(layers.Conv2D(
#         filters=hp.Choice('conv3_filters', [128, 256]),
#         kernel_size=hp.Choice('conv3_kernel', [3, 5]),
#         activation='relu'))
#
#     model.add(layers.Flatten())
#
#     # Dense Layer
#     model.add(layers.Dense(
#         units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
#         activation='relu'))
#
#     # Output
#     model.add(layers.Dense(4, activation='softmax'))
#
#     # Compile
#     model.compile(
#         optimizer=keras.optimizers.Adam(
#             hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy'])
#
#     return model
#
# min10, max10, min11, max11 = generate_global_max()
# load_data(min10, max10, min11, max11)
#
# X = np.array(X)
# y = np.array(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# X_train = X_train[..., np.newaxis]
# X_test = X_test[..., np.newaxis]
#
# tuner = kt.RandomSearch(
#         build_model,
#         objective='val_accuracy',
#         max_trials=20,
#         executions_per_trial=1,
#         directory='cnn_tuning',
#         project_name='palisade_cnn')
#
# tuner.search(X_train, y_train, epochs=10, validation_split=0.2,
#                  callbacks=[keras.callbacks.EarlyStopping(patience=3)])
#
# best_model = tuner.get_best_models(num_models=1)[0]
# tuner.results_summary()
