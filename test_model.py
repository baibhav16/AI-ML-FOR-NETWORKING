import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and scaler
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load the trained model
model = load_model("xss_blstm_cnn_with_features.h5")

# Maximum length used during training
max_len = 200

# === Feature Extraction Function ===
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

# === Predict Function ===
def predict_xss(text):
    # Deep learning input
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)

    # Engineered features
    features = extract_features(text)
    scaled_features = scaler.transform([list(features.values())])

    # Predict
    pred = model.predict([padded, scaled_features])[0][0]
    return "XSS Detected" if pred > 0.5 else "Safe"

# === Test ===
if __name__ == "__main__":
    sample_inputs = [
        '<tr><td class="plainlist" style="padding:0 0.1em 0.4em">',
        '<figcaption onpointerleave=alert(1)>XSS</figcaption>',
        '<li><a href="/wiki/Computer_data_storage" title="Computer data storage">Information storage systems </a> </li>',
        '<rtc draggable="true" ondragleave="alert(1)">test</rtc>',
    ]

    for text in sample_inputs:
        print(f"\nInput: {text}")
        print("Prediction:", predict_xss(text))
