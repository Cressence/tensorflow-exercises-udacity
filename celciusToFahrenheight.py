import tensorflow.compat.v1 as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celcius_q = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheight_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for index, count in enumerate(celcius_q) :
    print("{} degree Celcius = {} degree Fahrenheight".format(count, fahrenheight_a[index]))

layer = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([layer])

model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celcius_q, fahrenheight_a, epochs=500, verbose=False)
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
# plt.show()

print(model.predict([100.0]))
print("These are the layer variables: {}".format(layer.get_weights()))