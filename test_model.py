import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Load tokenizer ===
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# === Load trained model ===
model = load_model("xss_blstm_cnn_model.h5")

# === Max length (should match training) ===
max_len = 200

# === Prediction function ===
def predict_xss(input_text):
    sequence = tokenizer.texts_to_sequences([input_text])
    padded_seq = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_seq)[0][0]
    
    label = "XSS Attack ❌" if prediction >= 0.5 else "Safe ✅"
    print(f"Input: {input_text}")
    print(f"Prediction Score: {prediction:.4f} → {label}")
    print("-" * 60)

# === Test samples ===
samples = [
    "email@example.com"
]

# === Run predictions ===
for sample in samples:
    predict_xss(sample)
