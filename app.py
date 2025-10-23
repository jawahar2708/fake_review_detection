import json
import numpy as np
from scipy.sparse import hstack
from flask import Flask, render_template, request
import joblib
from xgboost import XGBClassifier

app = Flask(__name__)

# ===========================
# 🔹 Load All Models & Assets
# ===========================

# Logistic Regression, Random Forest, Naive Bayes, SVM bundles
log_reg_bundle = joblib.load("C:/Users/jawah/Desktop/fake review app/models/model_logreg.joblib")
rf_bundle = joblib.load("C:/Users/jawah/Desktop/fake review app/models/model_rf.joblib")
nb_bundle = joblib.load("C:/Users/jawah/Desktop/fake review app/models/model_nb.joblib")
svm_bundle = joblib.load("C:/Users/jawah/Desktop/fake review app/models/model_svm.joblib")

# XGBoost model + its TF-IDF and numeric feature info
xgb_tfidf = joblib.load("C:/Users/jawah/Desktop/fake review app/models/tfidf_xgb.joblib")

xgb_model = XGBClassifier()
xgb_model.load_model("C:/Users/jawah/Desktop/fake review app/models/model_xgb.json")

with open("C:/Users/jawah/Desktop/fake review app/models/xgb_numeric_cols.json") as f:
    xgb_numeric_cols = json.load(f)

# ==========================================================
# 🔹 Define Consistent XGBoost Prediction (Shape-Safe Version)
# ==========================================================
def predict_xgb(review_text):
    """Predict review authenticity using XGBoost + TF-IDF + numeric features."""
    review_dict = {
        'clean_text': review_text,
        'review_length': len(review_text.split()),
        'char_length': len(review_text),
        'exclamation_count': review_text.count('!'),
        'uppercase_ratio': sum(1 for c in review_text if c.isupper()) / max(len(review_text), 1)
    }

    # Transform with *the exact same fitted TF-IDF vectorizer*
    X_text = xgb_tfidf.transform([review_dict['clean_text']])

    # Get numeric features in correct order
    X_num = np.array([[review_dict[col] for col in xgb_numeric_cols]])

    # Combine TF-IDF + numeric
    X_comb = hstack([X_text, X_num])

    # Sanity check (optional, helps debugging)
    expected_features = xgb_model.get_booster().num_features()
    actual_features = X_comb.shape[1]
    if expected_features != actual_features:
        raise ValueError(f"⚠️ Feature shape mismatch: expected {expected_features}, got {actual_features}")

    # Predict
    pred = xgb_model.predict(X_comb)[0]
    return pred

# ===================================
# 🔹 Extract Other Classifiers & TF-IDF
# ===================================
log_reg, log_tfidf = log_reg_bundle["clf"], log_reg_bundle["tfidf"]
rf_model, rf_tfidf = rf_bundle["clf"], rf_bundle["tfidf"]
nb_model, nb_tfidf = nb_bundle["clf"], nb_bundle["tfidf"]
svm_model, svm_tfidf = svm_bundle["clf"], svm_bundle["tfidf"]

# ===============================
# 🔹 Other Model Prediction Helpers
# ===============================
def predict_logistic(text):
    X = log_tfidf.transform([text])
    return log_reg.predict(X)[0]

def predict_rf(text):
    X = rf_tfidf.transform([text])
    return rf_model.predict(X)[0]

def predict_nb(text):
    X = nb_tfidf.transform([text])
    return nb_model.predict(X)[0]

def predict_svm(text):
    X = svm_tfidf.transform([text])
    return svm_model.predict(X)[0]

# ===============================
# 🔹 Flask Route for Web Interface
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        review_text = request.form["review"]
        algorithm = request.form["algorithm"]

        if algorithm == "xgb":
            pred = predict_xgb(review_text)
        elif algorithm == "logistic":
            pred = predict_logistic(review_text)
        elif algorithm == "rf":
            pred = predict_rf(review_text)
        elif algorithm == "nb":
            pred = predict_nb(review_text)
        elif algorithm == "svm":
            pred = predict_svm(review_text)
        else:
            pred = 0  # default fallback

        result = "Fake Review" if pred == 1 else "Genuine Review"

    return render_template("index.html", result=result)

# ===============================
# 🔹 Run Flask App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
