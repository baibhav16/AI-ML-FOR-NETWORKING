import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# === Load dataset ===
df = pd.read_csv("dataset/XSS_dataset.csv")
texts = df['Sentence'].astype(str).values
labels = df['Label'].values

# === Manual Feature Engineering ===
def extract_features(text):
    return {
        'script_count': text.lower().count('<script'),
        'html_tag_count': len(re.findall(r"<[^>]+>", text)),
        'special_char_count': len(re.findall(r"[<>='\"/]", text)),
        'url_encoding_count': len(re.findall(r"%[0-9a-fA-F]{2}", text)),
        'length': len(text),
        'has_alert': int('alert(' in text.lower()),
        'has_on_event': int(bool(re.search(r'on\w+=', text.lower()))),
        'has_eval': int('eval(' in text.lower()),
        'has_iframe': int('<iframe' in text.lower())
    }

feature_dicts = [extract_features(t) for t in texts]
feature_df = pd.DataFrame(feature_dicts)
scaler = StandardScaler()
X_manual = scaler.fit_transform(feature_df)

# === Tokenization (char-level) ===
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_len = 200
X_seq = pad_sequences(sequences, maxlen=max_len)

# Save tokenizer and scaler
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# === Combine Deep + Manual Features ===
X = [X_seq, X_manual]
y = np.array(labels)

# === Train-test split ===
X_seq_train, X_seq_test, X_manual_train, X_manual_test, y_train, y_test = train_test_split(
    X_seq, X_manual, y, test_size=0.2, random_state=42
)

# === Model Inputs ===
seq_input = Input(shape=(max_len,), name='sequence_input')
feat_input = Input(shape=(X_manual.shape[1],), name='manual_features_input')

# === Sequence Model Branch ===
x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len)(seq_input)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Conv1D(64, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)

# === Manual Feature Branch ===
y_manual = Dense(32, activation='relu')(feat_input)

# === Combine Both Branches ===
combined = Concatenate()([x, y_manual])
combined = Dense(64, activation='relu')(combined)
combined = Dropout(0.5)(combined)
output = Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[seq_input, feat_input], outputs=output)

# === Compile Model ===
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# === Train Model ===
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
history = model.fit(
    [X_seq_train, X_manual_train],
    y_train,
    epochs=10,
    batch_size=4,
    validation_split=0.2,
    callbacks=[reduce_lr]
)

# === Save model ===
model.save("xss_blstm_cnn_with_features.h5")
print("[INFO] Model, tokenizer, and scaler saved.")

# === Plot accuracy & loss ===
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
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

# === Predict and Evaluate ===
y_pred_prob = model.predict([X_seq_test, X_manual_test])
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n[INFO] Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n[INFO] Classification Report:\n", classification_report(y_test, y_pred))
print("\n[INFO] Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
