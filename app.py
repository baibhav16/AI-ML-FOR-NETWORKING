import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("xss_blstm_cnn_model.h5")
max_len = 200
def predict_xss(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)
    predictions = model.predict(padded).flatten()
    results = []
    for text, prob in zip(texts, predictions):
        label = "❌ XSS Attack" if prob >= 0.5 else "✅ Safe"
        results.append({
            "Input": text,
            "Confidence": float(prob),
            "Prediction": label
        })
    return pd.DataFrame(results)


def render_bar(prob, is_attack):
    color = "red" if is_attack else "green"
    percent = int(prob * 100) if is_attack else int((1 - prob) * 100)
    st.markdown(f"""
    <div style="background-color:{color}; width:{percent}%; padding:4px; border-radius:5px; text-align:center; color:white; font-weight:bold;">
        {percent}% {('Malicious' if is_attack else 'Safe')}
    </div>
    """, unsafe_allow_html=True)


st.set_page_config(page_title="XSS Detector", layout="wide")
st.title("🛡️ XSS Attack Detector")
st.markdown("Paste multiple inputs or upload a file to check each line for XSS vulnerabilities.")

st.subheader("🔹 Paste inputs (one per line):")
user_input = st.text_area("", placeholder="<script>alert('XSS')</script>", height=150)


st.subheader("🔹 Or upload a .txt/.csv file:")
uploaded_file = st.file_uploader("", type=["txt", "csv"])

inputs = []


if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
        inputs = [line.strip() for line in content.splitlines() if line.strip()]
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        inputs = df.iloc[:, 0].dropna().astype(str).tolist()

if user_input.strip():
    inputs.extend([line.strip() for line in user_input.splitlines() if line.strip()])


if st.button("🔍 Analyze"):
    if not inputs:
        st.warning("Please enter or upload some inputs to analyze.")
    else:
        results_df = predict_xss(inputs)

        st.subheader("📋 Results")
        for i, row in results_df.iterrows():
            is_attack = row["Confidence"] >= 0.5
            st.markdown(f"**{i+1}. Input:**")
            st.code(row["Input"], language="html")
            st.markdown(f"**Prediction:** {row['Prediction']} (Confidence: `{row['Confidence']:.4f}`)")
            render_bar(row["Confidence"], is_attack)
            st.markdown("---")

        st.subheader("⬇️ Download Results as CSV")
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "xss_predictions.csv", "text/csv")


st.markdown("<hr style='margin-top:40px;'>", unsafe_allow_html=True)

