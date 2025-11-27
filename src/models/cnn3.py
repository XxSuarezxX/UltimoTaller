import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn3 import build_cnn3

# Configuración general
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 47

# Rutas a dataset (ajustar según tu estructura)
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', shuffle=False
)

# Construir modelo
model = build_cnn3(input_shape=IMG_SIZE+(3,), num_classes=NUM_CLASSES)
model.summary()

# Entrenamiento
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Graficar Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy por época')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("results/cnn3_accuracy.png")

# Graficar Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss por época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.savefig("results/cnn3_loss.png")

# Matriz de confusión en test
y_true = test_gen.classes
y_pred = model.predict(test_gen)
y_pred_classes = y_pred.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig("results/cnn3_confusion_matrix.png")
