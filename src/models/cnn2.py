import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn2(input_shape=(224,224,3), num_classes=47):
    inputs = layers.Input(shape=input_shape)
    
    # Normalización inicial
    x = layers.Rescaling(1./255)(inputs)
    
    # Bloque convolucional 1
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)  # Batch Normalization
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # Bloque convolucional 2
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # Bloque convolucional 3
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # Flatten + densas
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)  # Dropout opcional
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs, name="CNN2_Conv32_64_128")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
