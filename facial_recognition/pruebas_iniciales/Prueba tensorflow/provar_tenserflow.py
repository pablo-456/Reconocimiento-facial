from tensorflow import keras
import numpy as np

#Creas entorno virtual: python -m venv tf-env
#Iniciar entorno virtual: .\tf-env\Scripts\activate
#Nota: se deve instalar las herramientas a usar en el entorno

x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=10, verbose=0)

print("Predicción para 10:", model.predict(np.array([10.0])))

