import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate data
celsius = np.random.uniform(-50, 50, 1000)
fahrenheit = (celsius * 9/5) + 32
celsius = celsius.reshape(-1, 1)
fahrenheit = fahrenheit.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(celsius, fahrenheit, test_size=0.2, random_state=42)

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)

# Evaluate
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Train Loss: {train_loss}")
print(f"Test Loss: {test_loss}")

# Plot Training Curves
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Predict
celsius_temp = np.array([[100]])
predicted_fahrenheit = model.predict(celsius_temp)
print(f"Predicted Fahrenheit: {predicted_fahrenheit[0][0]}")