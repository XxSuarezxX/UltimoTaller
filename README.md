Taller de Inteligencia Computacional - Clasificación de Posturas de Yoga

Este repositorio contiene la implementación de un proyecto de clasificación de posturas de yoga mediante redes neuronales artificiales y convolucionales (MLP y CNNs). El objetivo del proyecto fue experimentar con diferentes arquitecturas de modelos, analizar su desempeño y documentar métricas clave como accuracy, loss y sobreajuste.

Importante: La versión original del proyecto no pudo ser subida a GitHub debido al tamaño del dataset y dependencias pesadas de librerías externas. Lo que se encuentra en este repositorio es una versión simplificada que permite reproducir la estructura del proyecto y entrenar los modelos sobre un conjunto de datos reducido o procesado.

Contenido

src/models/ → Contiene las definiciones de los modelos (MLP y CNNs).

src/train_*.py → Scripts para entrenar cada modelo y generar gráficos de loss y accuracy.

src/test_*.py → Scripts para visualizar la arquitectura de los modelos con model.summary().

dataset/ → Carpeta para colocar las imágenes procesadas (no incluida por tamaño).

results/ → Carpeta donde se guardan los gráficos de entrenamiento y matrices de confusión.

Arquitecturas implementadas

MLP Baseline: Red neuronal totalmente conectada con dos capas densas intermedias.

CNN1: Primera CNN sencilla con dos convoluciones, max pooling y dos capas densas.

CNN2: CNN más profunda con tres bloques convolucionales, Dropout y capas densas intermedias.

CNN3: CNN más compleja con múltiples bloques convolucionales, Batch Normalization, Dropout y capas densas.

Técnicas implementadas

Data Augmentation: Rotación, desplazamiento, volteo horizontal y zoom para mejorar la generalización.

Batch Normalization: Implementada en la arquitectura más compleja para estabilizar el entrenamiento.

Regularización con Dropout: Reducir sobreajuste en modelos profundos.

Evidencias documentadas

model.summary() completo con timestamp para cada arquitectura.

Gráficos de loss y accuracy (train/val) por época.

Matrices de confusión sobre el conjunto de prueba.

Tabla comparativa de resultados incluyendo accuracy, parámetros, tiempo por época y overfitting.
