import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load data
X_train = np.load("train_data.npy")
y_train = np.load("train_labels.npy")
X_val = np.load("val_data.npy")
y_val = np.load("val_labels.npy")

# Define Lightweight 1D CNN
model = models.Sequential([
    layers.Conv1D(16, kernel_size=5, activation="relu", input_shape=(23, 256)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(32, kernel_size=3, activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall(name="sensitivity")]
)

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

# Save model
model.save("seizure_cnn.h5")
print("Model training complete and saved as seizure_cnn.h5")