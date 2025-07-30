# Re-create the corrected app.py file after kernel reset
# ------------------- Imports -------------------
import streamlit as st
import torch
import shap
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os
import sqlite3
from io import BytesIO
from datetime import datetime
import uuid
import re
from opt_tcn_model import TCNWithAttention

# ------------------- Paths & Constants -------------------
MODEL_PATH = "tcn_best_model.pt"
TOKENIZER_PATH = "char2idx.pkl"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "phishing_logs.db")
MAX_LEN = 200
MAX_URLS = 1000

# ------------------- Session ID -------------------
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# ------------------- Sanitize -------------------
def sanitize_url(url):
        return re.sub(r'[\'\";<>{}]', '', url.strip())

# ------------------- DB Setup -------------------
def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                url TEXT,
                prediction TEXT,
                confidence REAL,
                feedback TEXT
            )
        ''')
        conn.commit()
        conn.close()

def log_to_db(url, prediction, confidence, feedback=""):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO logs (timestamp, session_id, url, prediction, confidence, feedback)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            st.session_state["session_id"],
            url,
            prediction,
            confidence,
            feedback
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Failed to write to database: {e}")

init_db()

# ------------------- Tokenizer -------------------
with open(TOKENIZER_PATH, "rb") as f:
    char2idx = pickle.load(f)

class CharTokenizer:
    def __init__(self, char2idx, max_length):
        self.char2idx = char2idx
        self.max_length = max_length
        self.pad_idx = 0

    def encode(self, url):
        encoded = [self.char2idx.get(c, self.pad_idx) for c in url]
        return (encoded + [self.pad_idx] * (self.max_length - len(encoded)))[:self.max_length]

    @property
    def vocab_size(self):
        return len(self.char2idx) + 1

tokenizer = CharTokenizer(char2idx, MAX_LEN)

# ------------------- Load Model -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCNWithAttention(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ------------------- SHAP -------------------
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
    plt.title(f"SHAP for: {url}\\nPrediction: {prediction} ({confidence:.4f})")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return prediction, confidence, buf

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="TCN Phishing Detector", layout="centered")
st.title("Real-Time Phishing Detector for Brand Protection")

st.markdown("""
<div style='background-color:#fff3cd; padding:10px; border-left:5px solid #ffc107;'>
    <b>Disclaimer:</b> This tool is for <i>educational and demonstration purposes only</i>.
</div>
""", unsafe_allow_html=True)

url_input = st.text_input("Enter a URL for prediction", "")
uploaded_file = st.file_uploader("Or upload a CSV/TXT file with URLs", type=['csv', 'txt'])

# ------------------- Single Prediction -------------------
if st.button("Predict for Single URL") and url_input:
    with st.spinner("Analyzing... please wait"):
        sanitized = sanitize_url(url_input)
        pred, conf, plot_buf = explain_url(sanitized)
        log_to_db(sanitized, pred, conf)
    st.success(f"Prediction: **{pred}** ({conf:.4f})")
    st.image(plot_buf, caption="SHAP Explanation", use_container_width=True)
    st.download_button("Download SHAP Plot", data=plot_buf, file_name=f"shap_{sanitized.replace('/', '_')}.png", mime="image/png")

    st.markdown("#### Was this prediction helpful or accurate?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes"):
            log_to_db(sanitized, pred, conf, feedback="yes")
            st.success("Thank you for your feedback!")
    with col2:
        if st.button("👎 No"):
            log_to_db(sanitized, pred, conf, feedback="no")
            st.info("Thanks! We'll keep improving.")

# ------------------- Batch Prediction -------------------
if uploaded_file is not None:
    if st.button("Predict for All URLs in File"):
        with st.spinner("Processing file... please wait"):
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                urls = df.iloc[:, 0].dropna().tolist()
            else:
                urls = uploaded_file.read().decode("utf-8").splitlines()

            if len(urls) > MAX_URLS:
                st.error(f"File contains {len(urls)} URLs, but limit is {MAX_URLS}.")
                st.stop()

            results = []
            for url in urls:
                sanitized = sanitize_url(url)
                pred, conf, _ = explain_url(sanitized)
                log_to_db(sanitized, pred, conf)
                results.append((sanitized, pred, conf))

            results_df = pd.DataFrame(results, columns=["URL", "Prediction", "Confidence"])

            def highlight(row):
                if row["Prediction"] == "Phishing":
                    return ['background-color: #ffe6e6; color: red'] * len(row)
                return ['background-color: #e6ffe6; color: green'] * len(row)

            st.dataframe(results_df.style.apply(highlight, axis=1))
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Prediction Results CSV", csv, "predictions.csv", "text/csv")

# ------------------- View Logs -------------------
if st.checkbox("Show Logs"):
    conn = sqlite3.connect(DB_PATH)
    df_logs = pd.read_sql("SELECT * FROM logs ORDER BY timestamp DESC", conn)
    conn.close()
    st.dataframe(df_logs)

