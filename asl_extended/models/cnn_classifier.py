import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_classifier(input_shape=(128, 128, 1), num_classes=27):
    model = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=input_shape),
        layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
        
        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
        
        layers.Flatten(),
        
        layers.Dense(units=128, activation='relu'),
        layers.Dropout(0.40),
        layers.Dense(units=96, activation='relu'),
        layers.Dropout(0.40),
        layers.Dense(units=64, activation='relu'),
        layers.Dense(units=num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_model(model_path_json, weights_path_h5):
    json_file = open(model_path_json, "r")
    model_json = json_file.read()
    json_file.close()
    
    model = keras.models.model_from_json(model_json)
    model.load_weights(weights_path_h5)
    return model

def save_model(model, json_path, weights_path):
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_path)

