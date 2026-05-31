"""
AI Medical Assistant With Severity Triage
# I am running this script on my Ubutnu
# Python 3.12.3
# Description:    Ubuntu 24.04.4 LTS
# Release:        24.04
# Codename:       noble

"""

import random
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

random.seed(42)

# sample data by disease and small explanation
data_map = {
    "Common Cold": "fever cough runny nose sore throat",
    "Flu": "fever body pain fatigue chills headache",
    "Migraine": "headache nausea light sensitivity",
    "Diabetes": "frequent urination thirst fatigue weight loss",
    "Hypertension": "headache dizziness chest pain",
    "Asthma": "shortness of breath wheezing cough",
    "Allergy": "sneezing itching skin rash",
    "Gastritis": "stomach pain nausea bloating"
}

# disease and severity
severity_map = {
    "Common Cold": "mild",
    "Flu": "moderate",
    "Migraine": "moderate",
    "Diabetes": "chronic",
    "Hypertension": "chronic",
    "Asthma": "moderate",
    "Allergy": "mild",
    "Gastritis": "moderate"
}
# disease and applicable medicines
medicine_map = {
    "Common Cold": ["Paracetamol", "Cetirizine"],
    "Flu": ["Paracetamol", "Ibuprofen"],
    "Migraine": ["Ibuprofen"],
    "Diabetes": ["Metformin"],
    "Hypertension": ["Amlodipine"],
    "Asthma": ["Salbutamol"],
    "Allergy": ["Cetirizine"],
    "Gastritis": ["Omeprazole"]
}


# Generate data set
X_text = []
y_disease = []
y_severity = []

for disease, symptoms in data_map.items():
    for _ in range(1000):
        noise = random.choice([
            "",
            " fatigue",
            " mild",
            " severe",
            " dizziness"
        ])

        X_text.append(symptoms + noise)
        y_disease.append(disease)
        y_severity.append(severity_map[disease])

df = pd.DataFrame({
    "text": X_text,
    "disease": y_disease,
    "severity": y_severity
})

df.to_csv("medical_dataset.csv", index=False)

print("Dataset generated")


# Tfidf VECTOR
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
# Label encoding
le_disease = LabelEncoder()
le_severity = LabelEncoder()

y_disease = le_disease.fit_transform(df["disease"])
y_severity = le_severity.fit_transform(df["severity"])

# xgb classifier
model_disease = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss"
)

model_severity = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    eval_metric="mlogloss"
)

model_disease.fit(X, y_disease)
model_severity.fit(X, y_severity)

print("Models trained")

# Save trained model pkl files
joblib.dump(model_disease, "model_disease.pkl")
joblib.dump(model_severity, "model_severity.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(le_disease, "le_disease.pkl")
joblib.dump(le_severity, "le_severity.pkl")

print("Model Saved")
print ("-"*50)

# Predict disease and its severity
def predict_disease_and_severity(symptoms):
    text = " ".join(symptoms)

    X_in = vectorizer.transform([text])

    d = model_disease.predict(X_in)[0]
    s = model_severity.predict(X_in)[0]

    disease = le_disease.inverse_transform([d])[0]
    severity = le_severity.inverse_transform([s])[0]

    return disease, severity

def get_meds(disease):
    return medicine_map.get(disease, [])

def triage(severity):
    return {
        "mild": "GREEN",
        "moderate": "YELLOW",
        "chronic": "RED"
    }.get(severity, "UNKNOWN")


if __name__ == "__main__":

    # test = ["fever", "cough", "runny nose"]
    test_inputs = [ 
        [ "cough", "fever"], 
        ["fever", "body pain", "fatigue", "chills"],
        ["headache", "dizziness"],
        ["sneezing", "itching"],
        ["nausea"],
        ["nausea", "bloating"] 
    ]
    import random
    rng = random.Random()
    test = rng.choice(test_inputs)

    disease, severity = predict_disease_and_severity(test)

    print("\n Input Symptoms:", test)
    print("\n Disease:", disease)
    print(" Severity:", severity)
    print(" Triage:", triage(severity))
    print(" Medicines:", get_meds(disease))