import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


data = load_breast_cancer()
print(data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(solver="lbfgs", max_iter=1000)
model.fit(X_train_scaled, y_train)

app = Flask(__name__)
CORS(app)

def create_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return plot_base64


@app.route("/predict", methods=["POST"])
def predict():
    features = request.json["features"]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[:, 1][0]
 
    # Calculate the probabilities for the test dataset
    y_scores = model.predict_proba(X_test_scaled)[:, 1]

    # Generate the ROC curve
    plot_base64 = create_roc_curve(y_test, y_scores)


    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability),
        "plot": plot_base64
    })


if __name__ == "__main__":
    app.run(debug=True)
