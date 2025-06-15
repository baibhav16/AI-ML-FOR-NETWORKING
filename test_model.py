import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model("xss_blstm_cnn_model.h5")
max_len = 200

# Function to predict XSS
def predict_xss(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0][0]
    return "Malicious (XSS)" if pred > 0.5 else "Benign"

# Example tests
samples = [
   "<script>alert('XSS')</script>",
    '<div onmouseover="alert(\'XSS\')">Hover me!</div>',
    '''<form action="/submit" method="POST">
    <input type="text" name="username">
    <button type="submit">Submit</button></form>''',
    '<a href="javascript:alert(\'XSS\')">Click me</a>'

]

for text in samples:
    result = predict_xss(text)
    print(f"Input: {text}\nPrediction: {result}\n")
