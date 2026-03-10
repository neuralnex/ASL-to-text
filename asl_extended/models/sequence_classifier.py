import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_bilstm_classifier(sequence_length=30, feature_dim=160, num_classes=50, 
                            lstm_units=[128, 64], dropout_rate=0.3):
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, feature_dim)),
        
        layers.Bidirectional(layers.LSTM(lstm_units[0], return_sequences=True)),
        layers.Dropout(dropout_rate),
        
        layers.Bidirectional(layers.LSTM(lstm_units[1], return_sequences=False)),
        layers.Dropout(dropout_rate),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate * 0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_gru_classifier(sequence_length=30, feature_dim=160, num_classes=50,
                          gru_units=[128, 64], dropout_rate=0.3):
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, feature_dim)),
        
        layers.Bidirectional(layers.GRU(gru_units[0], return_sequences=True)),
        layers.Dropout(dropout_rate),
        
        layers.Bidirectional(layers.GRU(gru_units[1], return_sequences=False)),
        layers.Dropout(dropout_rate),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model(model, json_path, weights_path):
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_path)

def load_model(json_path, weights_path):
    json_file = open(json_path, "r")
    model_json = json_file.read()
    json_file.close()
    
    model = keras.models.model_from_json(model_json)
    model.load_weights(weights_path)
    return model

