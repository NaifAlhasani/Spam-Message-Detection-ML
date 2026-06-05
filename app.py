# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import re

app = Flask(__name__)

# ===============================
# Load Vectorizer + Models
# ( ملفات التدريب Models)
# ===============================
vectorizer = joblib.load("tfidf3_vectorizer.pkl")
nb_model   = joblib.load("naive3_bayes_model.pkl")
svm_model  = joblib.load("svm3_model(2).pkl")

# ===============================
# Clean text (مثل التدريب)
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# Confidence
# ===============================
def get_confidence(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        return round(float(np.max(proba)) * 100, 1)

    if hasattr(model, "decision_function"):
        margin = float(model.decision_function(X)[0])
        conf = 1 / (1 + np.exp(-abs(margin)))
        return round(conf * 100, 1)

    return None

# ===============================
# Predict
# HAM = 0, SPAM = 1
# ===============================
def predict_message(msg, model_choice):
    msg_clean = clean_text(msg)
    X = vectorizer.transform([msg_clean])

    if model_choice == "nb":
        pred = int(nb_model.predict(X)[0])
        conf = get_confidence(nb_model, X)
        model_used = "Naive Bayes (Model 1)"
    else:
        pred = int(svm_model.predict(X)[0])
        conf = get_confidence(svm_model, X)
        model_used = "SVM (Model 2)"

    label_text = "SPAM" if pred == 0 else "HAM"
    return label_text, conf, model_used

# ===============================
# Route
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    msg = ""
    error = None
    model_used = None
    model_choice = "svm"  # default

    if request.method == "POST":
        msg = request.form.get("message", "").strip()
        model_choice = request.form.get("model_choice", "svm")

        if not msg:
            error = "Please enter a message first."
        else:
            result, confidence, model_used = predict_message(msg, model_choice)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        msg=msg,
        error=error,
        model_used=model_used,
        model_choice=model_choice
    )



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
