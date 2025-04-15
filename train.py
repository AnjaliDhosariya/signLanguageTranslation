from function import *
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
from collections import Counter

# Label mapping
label_map = {label: num for num, label in enumerate(actions)}

print("Loading Data")
# Load sequences and labels
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"), allow_pickle=True)
                window.append(res)
            except FileNotFoundError:
                print(f"Missing file: {action}/{sequence}/{frame_num}.npy")
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
print("Loaded data successfully")

# Prepare data
X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
print("Label distribution:", Counter(np.argmax(y, axis=1)))

# Shuffle and split
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Callbacks
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Build a cleaner model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train
history = model.fit(X_train, y_train, validation_split=0.1, epochs=100, callbacks=[tb_callback, early_stop], batch_size=32)

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')

# Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("Classification Report:\n", classification_report(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='Train Acc')
plt.plot(history.history['val_categorical_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()
