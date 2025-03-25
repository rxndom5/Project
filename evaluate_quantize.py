import tensorflow as tf
import numpy as np

# Load test data
X_test = np.load("test_data.npy")
y_test = np.load("test_labels.npy")

# Load trained model
model = tf.keras.models.load_model("seizure_cnn.h5")

# Evaluate
results = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}, Sensitivity: {results[2]:.4f}")

# Quantize model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

# Save quantized model
with open("seizure_cnn.tflite", "wb") as f:
    f.write(tflite_model)
print("Quantized model saved as seizure_cnn.tflite")