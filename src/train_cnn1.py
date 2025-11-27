# src/train_cnn1.py
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn1 import build_cnn1

# --- Configuración ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Directorios del dataset (modifica según tu estructura)
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# --- Data Augmentation ---
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rescale=1./255
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', shuffle=False
)

# --- Construir modelo ---
model = build_cnn1(input_shape=(224,224,3), num_classes=train_gen.num_classes)
model.summary()

# --- Entrenamiento ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# --- Guardar modelo ---
model.save("results/cnn1_model.h5")

# --- Gráficos ---
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy por época")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss por época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("results/cnn1_history.png")
plt.show()

# --- Matriz de confusión ---
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_gen.class_indices.keys()))
disp.plot(xticks_rotation='vertical')
plt.title("Matriz de confusión - CNN1")
plt.savefig("results/cnn1_confusion_matrix.png")
plt.show()
