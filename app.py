import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Load tokenizer and model ===
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("xss_blstm_cnn_model.h5")
max_len = 200

# === Streamlit UI ===
st.title("🛡️ XSS Attack Detector")
st.write("Enter any text input (like form data, script, or HTML) to check for potential XSS attacks.")

user_input = st.text_area("📝 Input", height=150)

if st.button("🔍 Check"):
    if not user_input.strip():
        st.warning("Please enter some input text.")
    else:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(sequence, maxlen=max_len)

        # Predict
        prediction = model.predict(padded_seq)[0][0]
        label = "❌ XSS Attack Detected!" if prediction >= 0.5 else "✅ Safe Input"

        # Show result
        st.subheader(label)
        st.caption(f"Model Confidence: `{prediction:.4f}`")
