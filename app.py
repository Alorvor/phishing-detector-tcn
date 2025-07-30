import streamlit as st
import torch
import shap
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os
from io import BytesIO
from datetime import datetime
from opt_tcn_model import TCNWithAttention

# Paths
MODEL_PATH = r"C:\Users\drmar\Documents\Capstone Project\TCN\tcn_best_model.pt"
TOKENIZER_PATH = r"C:\Users\drmar\Documents\Capstone Project\TCN\char2idx.pkl"
MAX_LEN = 200
LOG_PATH = "streamlit_log.csv"

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    char2idx = pickle.load(f)

class CharTokenizer:
    def __init__(self, char2idx, max_length):
        self.char2idx = char2idx
        self.max_length = max_length
        self.pad_idx = 0

    def encode(self, url):
        encoded = [self.char2idx.get(c, self.pad_idx) for c in url]
        if len(encoded) < self.max_length:
            encoded += [self.pad_idx] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
        return encoded

    @property
    def vocab_size(self):
        return len(self.char2idx) + 1

tokenizer = CharTokenizer(char2idx=char2idx, max_length=MAX_LEN)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCNWithAttention(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

class WrappedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.long).to(device)
        with torch.no_grad():
            logits, _ = self.base_model(x)
        return logits

wrapped_model = WrappedModel(model)
explainer = shap.Explainer(wrapped_model, shap.maskers.Independent(np.zeros((1, MAX_LEN), dtype=np.int64)))

# SHAP Explanation Function
def explain_url(url):
    encoded = tokenizer.encode(url)
    input_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(input_tensor)
        confidence = torch.sigmoid(logits).item()
        prediction = "Phishing" if confidence > 0.5 else "Legitimate"

    shap_values = explainer(input_tensor.cpu().numpy())
    values = shap_values.values[0]
    characters = list(url)
    colors = ['red' if v > 0 else 'blue' for v in values[:len(characters)]]

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(characters)), values[:len(characters)], color=colors)
    plt.xticks(range(len(characters)), characters)
    plt.xlabel("Character")
    plt.ylabel("SHAP Value")
    plt.title(f"SHAP for: {url}\nPrediction: {prediction} ({confidence:.4f})")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    return prediction, confidence, buf

# Logging
def log_result(url, prediction, confidence):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "url": url,
        "prediction": prediction,
        "confidence": confidence
    }
    if not os.path.exists(LOG_PATH):
        pd.DataFrame([entry]).to_csv(LOG_PATH, index=False)
    else:
        pd.concat([pd.read_csv(LOG_PATH), pd.DataFrame([entry])], ignore_index=True).to_csv(LOG_PATH, index=False)

# --------------- Streamlit UI ----------------
st.set_page_config(page_title="TCN Phishing Detector", layout="centered")
st.title(" Real-Time Phishing URL Detector (TCN + SHAP)")

# URL Input
url_input = st.text_input(" Enter a URL for prediction", "")

# File Upload
uploaded_file = st.file_uploader("input Or upload a CSV/TXT file with URLs", type=['csv', 'txt'])

# Process Single URL
if st.button(" Predict for Single URL") and url_input:
    pred, conf, plot_buf = explain_url(url_input)
    log_result(url_input, pred, conf)
    st.success(f"Prediction: **{pred}** ({conf:.4f})")
    st.image(plot_buf, caption="SHAP Explanation", use_column_width=True)
    st.download_button(" Download SHAP Plot", data=plot_buf, file_name=f"shap_{url_input.replace('/', '_')}.png", mime="image/png")

# Process Batch File
if uploaded_file is not None:
    if st.button(" Predict for All URLs in File"):
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            urls = df.iloc[:, 0].dropna().tolist()
        else:
            urls = uploaded_file.read().decode("utf-8").splitlines()

        results = []
        for url in urls:
            pred, conf, _ = explain_url(url)
            log_result(url, pred, conf)
            results.append((url, pred, conf))

        results_df = pd.DataFrame(results, columns=["URL", "Prediction", "Confidence"])
        st.dataframe(results_df)

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(" Download Prediction Results CSV", csv, "predictions.csv", "text/csv")

# View Log
if st.checkbox(" Show Log CSV"):
    if os.path.exists(LOG_PATH):
        st.dataframe(pd.read_csv(LOG_PATH))
    else:
        st.warning("No logs yet.")

