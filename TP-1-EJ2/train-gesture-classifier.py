import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Cargar los datos del archivo .npy
dataset = np.load('rps_dataset.npy', allow_pickle=True)
# import pdb; pdb.set_trace()

# Separar las coordenadas y etiquetas
coordinates = np.array([np.array(row[2])[:, :2].flatten() for row in dataset])
labels = np.array([row[1] for row in dataset])

# Preprocesar los datos
X = coordinates
y = to_categorical(labels, num_classes=3)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Red Neuronal
model = Sequential([
    Dense(64, activation='relu', input_shape=(42,1)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Guardar el modelo en un archivo .h5
model.save('rps_model.h5')

print("Modelo guardado en 'rps_model.h5'.")

# model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# # Evaluar el modelo
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Accuracy en datos de prueba: {test_accuracy:.2f}")
