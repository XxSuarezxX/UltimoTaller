# src/models/cnn1.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn1(input_shape=(224,224,3), num_classes=47):
    inputs = layers.Input(shape=input_shape)
    
    # Normalización simple de los píxeles
    x = layers.Rescaling(1./255)(inputs)
    
    # Bloque convolucional 1
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    
    # Bloque convolucional 2
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    
    # Reducción espacial
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # Aplanar características para las capas densas
    x = layers.Flatten()(x)
    
    # Capas densas intermedias
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    
    # Capa de salida
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # Crear modelo
    model = models.Model(inputs, outputs, name="CNN1_Conv32_64")
    
    # Compilar modelo
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
