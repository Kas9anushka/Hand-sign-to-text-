import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import os

# Assuming actions, DATA_PATH, no_sequences, and sequence_length are defined elsewhere
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

# Prepare data (sequences and labels)
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# TensorBoard callback
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))  # Adjust input_shape as needed
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Assuming actions is a list of class names

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model and capture the history
history = model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))

# Plot the accuracy
plt.figure(figsize=(10, 6))

# Plot training accuracy
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')

# Plot validation accuracy
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')

# Add labels and title
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Save model architecture and weights
model_json = model.to_json()
with open("model.json", "w", encoding="utf-8") as json_file:
    json_file.write(model_json)
model.save('model.h5')
