import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("dataset/XSS_dataset.csv")
texts = df['Sentence'].astype(str).values
labels = df['Label'].values

# tokenization for feature extraction in cnn model
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


max_len = 200
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Build CNN + BLSTM Model ===
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Conv1D(64, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


reduce_lr = ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)


history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=4,
    validation_split=0.2,
    callbacks=[reduce_lr]
)
# savving model as --
model.save("xss_blstm_cnn_model.h5")
print("[INFO] Model and tokenizer saved.")

plt.figure(figsize=(12, 5))

# training and validationg ploting graph for 25 epochs
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_validation_plot.png")  
plt.show()



y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# confusion matrix for accuracy check
print("\n[INFO] Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n[INFO] Classification Report:\n", classification_report(y_test, y_pred))
print("\n[INFO] Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
