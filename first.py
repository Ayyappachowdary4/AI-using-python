# -----------------------------------------------------------
# AI Project in Python 
# Deep Learning: Handwritten Digit Recognition (MNIST)
# -----------------------------------------------------------

# 1. Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os

# 2. Load MNIST dataset (digits 0-9)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 3. Explore dataset
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
print("Example label:", y_train[0])

# 4. Visualize a few samples
plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.suptitle("Sample MNIST Digits")
plt.show()

# 5. Preprocess data: normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images for dense network
x_train_flat = x_train.reshape((x_train.shape[0], -1))
x_test_flat = x_test.reshape((x_test.shape[0], -1))

# 6. One-hot encode labels
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# 7. Build the neural network model
model = models.Sequential([
    layers.Input(shape=(784,)),  # 28x28 flattened
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # output layer (10 digits)
])

# 8. Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 9. Print model summary
model.summary()

# 10. Train the model
history = model.fit(
    x_train_flat, y_train_cat,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# 11. Evaluate the model
test_loss, test_acc = model.evaluate(x_test_flat, y_test_cat, verbose=0)
print("\nTest Accuracy: {:.2f}%".format(test_acc * 100))
print("Test Loss:", test_loss)

# 12. Plot accuracy and loss graphs
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 13. Predict on a few test samples
predictions = model.predict(x_test_flat[:10])
predicted_labels = np.argmax(predictions, axis=1)

# 14. Show predicted vs actual
plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {predicted_labels[i]}, True: {y_test[i]}")
    plt.axis('off')
plt.suptitle("Predictions vs Actual Labels")
plt.show()

# 15. Predict on custom input
custom_digit = x_test[1]  # just reuse one from dataset
plt.imshow(custom_digit, cmap='gray')
plt.title("Custom Input Example")
plt.axis('off')
plt.show()

custom_input = custom_digit.reshape(1, 784)
custom_pred = model.predict(custom_input)
predicted_number = np.argmax(custom_pred)
print("\nAI Prediction for custom image:", predicted_number)

# 16. Save model to file
model_dir = "saved_model_mnist"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(os.path.join(model_dir, "digit_classifier.h5"))
print("\nModel saved successfully!")

# 17. Load model and re-evaluate
loaded_model = keras.models.load_model(os.path.join(model_dir, "digit_classifier.h5"))
re_loss, re_acc = loaded_model.evaluate(x_test_flat, y_test_cat, verbose=0)
print("Reloaded Model Accuracy: {:.2f}%".format(re_acc * 100))

# 18. Predict again using loaded model
sample = x_test[3].reshape(1, 784)
loaded_pred = np.argmax(loaded_model.predict(sample))
print("Prediction from loaded model:", loaded_pred)

# 19. End of program
print("\n Deep Learning AI Completed Successfully! ")
