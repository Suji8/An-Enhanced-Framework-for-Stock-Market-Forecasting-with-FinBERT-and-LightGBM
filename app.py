from pickle import APPEND


python APPEND.py# =========================================================
# app.py — Flask Web App for Stock Forecast using FinBERT + LightGBM
# =========================================================
from flask import Flask, render_template, request
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from collections import deque

# ---------------------------------------------------------
# 1. Initialize Flask App
# ---------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------
# 2. Load FinBERT Model (for sentiment)
# ---------------------------------------------------------
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
finbert = AutoModelForSequenceClassification.from_pretrained(model_name)
finbert.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert.to(device)

# ---------------------------------------------------------
# 3. Load LightGBM Model
# ---------------------------------------------------------
lgb_model = joblib.load("lightgbm_model.pkl")
# Store last 5 predictions
recent_predictions = deque(maxlen=5)

# ---------------------------------------------------------
# 4. Helper: Compute Sentiment Score using FinBERT
# ---------------------------------------------------------
def get_finbert_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = finbert(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # [negative, neutral, positive]
        score = probs[0][2].item() - probs[0][0].item()
    return score

# ---------------------------------------------------------
# 5. Home Page Route
# ---------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# ---------------------------------------------------------
# 6. Prediction Route
# ---------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news_text']
    sentiment_score = get_finbert_sentiment(news)
    sentiment_norm = (sentiment_score + 1) / 2
    features = np.array([[sentiment_score, sentiment_norm]])
    prediction = lgb_model.predict(features)[0]
    result = "📈 Market Up (Bullish)" if prediction == 1 else "📉 Market Down (Bearish)"

    # store history
    recent_predictions.append(f"{result} — Sentiment: {round(sentiment_score, 2)}")

    return render_template(
        'index.html',
        prediction=result,
        news_input=news,
        sentiment=round(sentiment_score, 2),
        history=list(recent_predictions)[::-1]  # reverse for latest first
    )

# ---------------------------------------------------------
# 7. Run App
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
