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

<b>Ejercicio 2:</b>

<b>Ejercicio 3:</b>
El objetivo de este problema es construir y comparar el rendimiento de distintos modelos de clasificación de imágenes utilizando redes neuronales convolucionales y densas que puedan clasificar con precisión las imágenes de escenas naturales de las seis categorías distintas, utilizando el dataset proporcionado.

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
