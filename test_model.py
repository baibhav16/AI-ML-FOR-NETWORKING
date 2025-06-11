import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model("xss_cnn_model.h5")
max_len = 200

# Function to predict XSS
def predict_xss(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0][0]
    return "Malicious (XSS)" if pred > 0.5 else "Benign"

# Example tests
samples = [
    "	 </span> <span class=""reference-text""><cite class=""citation news""><a rel=""nofollow"" class=""external text"" href=""https://www.economist.com/news/special-report/21700756-artificial-intelligence-boom-based-old-idea-modern-twist-not"">""From not working to neural networking"" </a>. <i>The Economist </i>. 2016<span class=""reference-accessdate"">. Retrieved <span class=""nowrap"">26 April ",
    "<caption id=x tabindex=1 ondeactivate=alert(1)></caption><input id=y autofocus>"

]

for text in samples:
    result = predict_xss(text)
    print(f"Input: {text}\nPrediction: {result}\n")
