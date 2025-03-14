# 1. Defining the data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow as tf
import matplotlib.pyplot as plt

# XOR dataset
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# 2. Creating the model
model = Sequential()
model.add(InputLayer(input_shape=(2,)))  # Input layer
model.add(Dense(2, activation='sigmoid'))  # Hidden layer with 2 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron

# 3. Compile
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)  # SGD optimizer with learning rate
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 4. Train the model
history = model.fit(X, y, epochs=2000, batch_size=1, verbose=0)  # Train for 2000 epochs

# 5. Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

# 6. Print predictions
for id_x, data_sample in enumerate(X):
    prediction = model.predict([data_sample], verbose=0)
    print(f"Data sample: {data_sample}, prediction: {prediction}, ground truth: {y[id_x]}")

# 7. Plot the loss
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.show()

# TASK: Play with hyperparameters
# Example: Changing number of neurons, learning_rate, epochs, batch_size, and activation functions
# Below are examples of experiments with different hyperparameters

# Change learning rate, neurons, and activation function
model = Sequential()
model.add(InputLayer(input_shape=(2,)))
model.add(Dense(4, activation='relu'))  # Changed to 4 neurons and relu activation
model.add(Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)  # Higher learning rate
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X, y, epochs=1000, batch_size=4, verbose=1)  # Changed epochs, batch_size, verbose

loss, accuracy = model.evaluate(X, y, verbose=0)
print('Experimented Model Accuracy: {:.2f}%'.format(accuracy * 100))

plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over epochs (Experimented Model)')
plt.show()

# Answer: We observe that changes in learning rate, number of neurons, and activation functions impact training stability and accuracy.

# EXERCISE 2

# 1. Import the necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt

# 2. Load the dataset
path_to_dataset = r"C:\Users\ivane\OneDrive\Bureau\ML-exercices-Ivan-BAJKIC/voting_complete.csv"  # Change the PATH
pd_dataset = pd.read_csv(path_to_dataset)

# 3. Define a function for train and test split
def train_test_split(pd_data: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    pd_dataset = pd_data.copy()
    pd_dataset = pd_dataset[pd_dataset.columns[1:]]
    index = np.arange(len(pd_dataset))
    index = np.random.permutation(index)

    train_ammount = int(len(index) * test_ratio)
    train_ids = index[train_ammount:]
    test_ids = index[:train_ammount]

    train_dataset = pd_dataset[pd_dataset.index.isin(train_ids)].reset_index()
    test_dataset = pd_dataset[pd_dataset.index.isin(test_ids)].reset_index()

    train_dataset = train_dataset[train_dataset.columns[1:]]
    test_dataset = test_dataset[test_dataset.columns[1:]]

    return train_dataset[train_dataset.columns[1:]], train_dataset[train_dataset.columns[0]], \
           test_dataset[test_dataset.columns[1:]], test_dataset[test_dataset.columns[0]]

# 4. Data processing
# Replace 'y' with 1, 'n' with 0, and '?' with np.nan
pd_dataset.replace({'y': 1, 'n': 0, '?': np.nan}, inplace=True)

# Fill missing values with mode (most frequent value) for each column
for column in pd_dataset.columns[1:]:
    pd_dataset[column].fillna(pd_dataset[column].mode()[0], inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(pd_dataset)

# Encode labels: republican -> 1, democrat -> 0
y_train = y_train.replace({'republican': 1, 'democrat': 0})
y_test = y_test.replace({'republican': 1, 'democrat': 0})

# 5. CREATING THE MODEL

# 1. create the model
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],)))  # Hidden layer with 12 neurons
model.add(Dense(8, activation='relu'))  # Hidden layer with 8 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# 2. print the model summary
model.summary()

# What does model.summary() do?
# It prints the model architecture, including each layer, its shape, number of parameters, and how many parameters are trainable.

# 3. compile
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 4. train the model
from sklearn.model_selection import train_test_split

# Additional split to get a validation set
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)

history = model.fit(X_train_split, y_train_split,
                    epochs=40,
                    batch_size=4,
                    verbose=1,
                    validation_data=(X_val, y_val))

# 6. Evaluate the model
# 7. MODEL EVALUATION

# 1. Preprocess test set
X_test.replace('?', np.nan, inplace=True)  # Just in case, though we already filled missing values earlier
X_test.fillna(X_test.mode().iloc[0], inplace=True)

# Convert categorical columns (not necessary if already numeric, just matching your example)
x_te = pd.get_dummies(X_test)

# Replace labels in y_test (done earlier, but keeping the structure)
y_te = y_test.replace({'republican': 1, 'democrat': 0})

# 2. Evaluate on test set
loss, accuracy = model.evaluate(x_te, y_te, verbose=0)
print('Accuracy: {:.2f}'.format(accuracy * 100))
print('Loss: {:.2f}'.format(loss * 100))

# 3. Plot training/validation loss and accuracy
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_ylabel('Value')
ax.set_xlabel('Epoch')
ax.set_title('Model Loss over Epochs')
ax.legend()
plt.show()

fig, ay = plt.subplots()
ay.plot(history.history['accuracy'], label='Training Accuracy')
ay.plot(history.history['val_accuracy'], label='Validation Accuracy')
ay.set_ylabel('Value')
ay.set_xlabel('Epoch')
ay.set_title('Model Accuracy over Epochs')
ay.legend()
plt.show()
