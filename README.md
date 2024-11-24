<p align="center">
  <img src="unr.logo.png" alt="Logo del proyecto" width="200"/>
</p>
<p align="center" style="font-size:24px;">
  <b>Trabajo Práctico 1</b>
</p> 
<p align="center" style="font-size:24px;">
  <b>Redes Densas y Convolucionales</b>
</p> 
<p align="center" style="font-size:24px;">
  <b>Aprendizaje Automático II - 2C 2024</b>
</p> 
<p align="center">
  Autores: Julián García - Lara Valeri
</p>

<b>Ejercicio 1:</b>
En este problema, se presenta un conjunto de datos que contiene información sobre el rendimiento académico de estudiantes universitarios,
así como diversos factores que podrían influir en él. El objetivo es construir un modelo de regresión utilizando redes neuronales para
predecir el índice de rendimiento académico de los estudiantes basado en las características proporcionadas.

Dataset: https://www.kaggle.com/c/titanic/overview

El dataset proporcionado incluye las siguientes variables para cada estudiante:

- Hours Studied: El número total de horas dedicadas al estudio por cada estudiante.
- Previous Scores: Las puntuaciones obtenidas por los estudiantes en exámenes previos.
- Extracurricular Activities: Si el estudiante participa en actividades extracurriculares (Sí o No).
- Sleep Hours: El número promedio de horas de sueño que el estudiante tuvo por día.
- Sample Question Papers Practiced: El número de cuestionarios de muestra que el estudiante practicó.
  Además, el dataset incluye la variable objetivo:
- Performance Index: Un índice que representa el rendimiento académico general de cada estudiante, redondeado al entero más cercano.
  Este índice varía de 10 a 100, donde valores más altos indican un mejor rendimiento.

La resolución de este ejercicio se encuentra en el archivo: <b>TP1-AAII-2C-2024-EJ1</b>. El dataset correspondiente se encuentra en el archivo <b>Student_Performance.csv</b>

<b>Ejercicio 2:</b>

El objetivo de este ejercicio es implementar un sistema de clasificación de gestos de "piedra", "papel" o "tijeras" utilizando MediaPipe para la detección de las manos y una red neuronal densa para realizar la clasificación. 

Este ejercicio cuenta con tres scripts que se encuentran dentro de la carpeta <b>TP-1-EJ2</b>. Se detalla a continuación la función que cumple cada uno:
- record-dataset.py: permitirá grabar un dataset de gestos utilizando la cámara web y MediaPipe para detectar los landmarks (puntos clave) de la mano. Cada gesto se etiquetará como "piedra" (0), "papel" (1) o "tijeras" (2) y se almacenará junto con sus coordenadas en archivos .npy.
- rock-paper-scissors.py: se entrenará una red neuronal densa utilizando los datos de los landmarks obtenidos en la primera parte. El modelo resultante será capaz de clasificar los gestos basándose en las posiciones de los puntos clave de la mano.
- train-gesture-classifier.py tomará como entrada la imagen de la cámara web, utilizará MediaPipe para detectar los landmarks de la mano, y clasificará el gesto en "piedra", "papel" o "tijeras" utilizando el modelo entrenado.

Además dentro de dicha carpeta se encuentra la carpeta <b>Imágenes ejercicio 2</b> que contiene imágenes donde se muestra el funcionamiento del sistema.

<b>Ejercicio 3:</b>
El objetivo de este problema es construir y comparar el rendimiento de distintos modelos de clasificación de imágenes utilizando redes
neuronales convolucionales y densas que puedan clasificar con precisión las imágenes de escenas naturales de las seis categorías
distintas, utilizando el dataset proporcionado.

El dataset proporcionado contiene alrededor de 25,000 imágenes de tamaño 150x150, distribuidas en seis categorías:

- buildings
- forest
- glacier
- mountain
- sea
- street
  Las imágenes están divididas en tres conjuntos:
- Train: Alrededor de 14,000 imágenes para entrenamiento.
- Test: Alrededor de 3,000 imágenes para evaluación del modelo.
- Prediction: Alrededor de 7,000 imágenes para predicción final.

Los modelos que diseñamos en este ejercicio son:

- Modelo con capas densas.
- Modelo con capas convolucionales y densas.
- Modelo que incluya bloques residuales identidad
- Modelo que utilice como backbone alguna de las arquitecturas disponibles en TensorFlow (transfer learning) --> Implementamos MobileNet

La reseloución de este ejercicio se encuentra en el archivo: <b>TP1-AAII-2C-2024-EJ3</b>. En la carpeta <b>TP-1-EJ3</b> se encuentran los dataset proporcionados para realiazar el mismo
