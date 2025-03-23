from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 Dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocessing: Scaling pixel values between 0 and 1
X_train_scaled = X_train.astype('float32') / 255.0
X_test_scaled = X_test.astype('float32') / 255.0

# Convert class labels to one-hot encoding
y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

# Create the neural network model
model = Sequential()

# Flatten layer to convert 3D image input (32x32x3) into 1D vector
model.add(Flatten(input_shape=(32, 32, 3)))

# Added more neurons (512) to increase model capacity
# ReLU activation works better with deep networks compared to sigmoid
model.add(Dense(512, activation='relu'))

# Dropout layer with 30% dropout rate to prevent overfitting
model.add(Dropout(0.3))

# Added another hidden layer with 256 neurons to increase network depth and capacity
model.add(Dense(256, activation='relu'))

# Dropout layer to further help with regularization and reduce overfitting
model.add(Dropout(0.3))

# Added a third hidden layer with 128 neurons to allow deeper feature extraction
model.add(Dense(128, activation='relu'))

# Additional Dropout for regularization
model.add(Dropout(0.3))

# Fourth hidden layer with 64 neurons to continue refining learned features
model.add(Dense(64, activation='relu'))

# Output layer with 10 neurons (one per class) and softmax activation to get class probabilities
model.add(Dense(10, activation='softmax'))

# Compile the model
# Changed optimizer from SGD to Adam because Adam generally works better and faster
# Learning rate set to 0.01 (increased from the default) to speed up convergence
optimizer = Adam(learning_rate=0.01)

model.compile(
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    optimizer=optimizer,
    metrics=['accuracy']  # Track accuracy during training and testing
)

# Display the model architecture
model.summary()

# Added EarlyStopping to stop training early if validation loss doesn't improve for 5 epochs
# This prevents overfitting and saves training time
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True  # Restore model weights from the epoch with the best validation loss
)

# Train the model
history = model.fit(
    X_train_scaled,
    y_train_encoded,
    epochs=50,            # Increased number of epochs to 50 to give the model time to learn
    batch_size=64,        # Reduced batch size from 128 to 64 to help the model generalize better
    validation_split=0.2, # Use 20% of the training data for validation
    callbacks=[early_stopping],  # Apply early stopping during training
    verbose=1
)

# Evaluate the model on the test dataset
score = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
print(f'\nTest loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]*100:.2f}%')

# Plot the training history to show loss and accuracy over epochs
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Function to visualize the best predictions made by the model
def show_the_best_predictions(model, x_test: np.array, y_test: np.array, n_of_pred: int = 10) -> None:
    mapping = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
               8: 'ship', 9: 'truck'}

    # Make predictions
    predictions = model.predict(x_test)

    # Reshape y_test for comparison
    y_test = y_test.reshape(1, -1)
    predictions_ind = np.argmax(predictions, axis=1).reshape(1, -1)

    # Find correct predictions
    correct_predictions = np.where(predictions_ind == y_test)
    rows_correct = correct_predictions[1]

    # Get probabilities and corresponding images for correct predictions
    predicted_probs = predictions[rows_correct]
    target_correct = y_test[0][rows_correct]

    # Select the best predictions (highest confidence)
    max_samples = predicted_probs[np.arange(len(rows_correct)), target_correct]
    selected_images = x_test[rows_correct]

    sorted_ind = np.argsort(max_samples)[::-1]

    images = []
    probs = []
    labels = []

    for ind in range(min(n_of_pred, len(sorted_ind))):
        index = sorted_ind[ind]
        labels.append(target_correct[index])
        probs.append(max_samples[index])
        images.append(selected_images[index])

    plt.figure(figsize=(20, 10))
    images_concat = np.concatenate(np.asarray(images), axis=1)
    plt.imshow(images_concat)

    for i in range(len(images)):
        text = f"{mapping[labels[i]]}:\n{probs[i]*100:.2f}%"
        plt.text((32 / 2) + 32 * i - len(mapping[labels[i]]), 32 * (5 / 4), text)

    plt.axis('off')
    plt.show()


show_the_best_predictions(model, X_test_scaled, y_test)
