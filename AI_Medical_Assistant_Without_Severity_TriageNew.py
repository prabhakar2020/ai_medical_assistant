"""
AI Medical Assistant Without Severity Triage
# I am running this script on my Ubutnu
# Python 3.12.3
# Description:    Ubuntu 24.04.4 LTS
# Release:        24.04
# Codename:       noble

"""

import random
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from xgboost import XGBClassifier

random.seed(42)

# Disease master data
diseases = [
    ("D001", "Common Cold", "mild"),
    ("D002", "Flu", "moderate"),
    ("D003", "Migraine", "moderate"),
    ("D004", "Diabetes", "chronic"),
    ("D005", "Hypertension", "chronic"),
    ("D006", "Asthma", "moderate"),
    ("D007", "Allergy", "mild"),
    ("D008", "Gastritis", "moderate"),
]

# Symptoms master data
symptom_bank = {
    "Common Cold": ["fever", "cough", "runny nose", "sore throat"],
    "Flu": ["fever", "body pain", "fatigue", "chills"],
    "Migraine": ["headache", "nausea", "light sensitivity"],
    "Diabetes": ["frequent urination", "thirst", "fatigue"],
    "Hypertension": ["headache", "dizziness", "chest discomfort"],
    "Asthma": ["shortness of breath", "wheezing", "cough"],
    "Allergy": ["sneezing", "itching", "skin rash"],
    "Gastritis": ["stomach pain", "nausea", "bloating"]
}

# Medicine mapping
medicine_map = {
    "Common Cold": ["Paracetamol", "Cetirizine"],
    "Flu": ["Paracetamol", "Ibuprofen"],
    "Migraine": ["Ibuprofen"],
    "Diabetes": ["Metformin"],
    "Hypertension": ["Amlodipine"],
    "Asthma": ["Salbutamol"],
    "Allergy": ["Cetirizine"],
    "Gastritis": ["Ibuprofen"]
}

# Generate Synthetic dataset
def generate__synthetic_data(n=6000):
    data = []

    for _ in range(n):
        disease = random.choice(diseases)
        d_name = disease[1]
        severity = disease[2]

        symptoms = symptom_bank[d_name]
        selected = random.sample(symptoms, random.randint(2, len(symptoms)))

        # noise injection
        if random.random() < 0.3:
            selected.append(random.choice(["fatigue", "headache", "dizziness"]))

        data.append({
            "symptoms": "|".join(selected),
            "disease": d_name,
            "severity": severity
        })

    return pd.DataFrame(data)

df = generate__synthetic_data()
df.to_csv("medical_dataset.csv", index=False)

print("Dataset generated")


# Encoding symptoms
df["symptoms"] = df["symptoms"].apply(lambda x: x.split("|"))

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptoms"])

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["disease"])


# Train model using xgbboost
model = XGBClassifier(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss"
)

model.fit(X, y)

# Save trained model pkl file
joblib.dump(model, "disease_model.pkl")
joblib.dump(mlb, "mlb.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model trained & saved")

# Extract medicines based on disease name
def get_extract_medicines(disease_name):
    return medicine_map.get(disease_name, [])

# predict_disease_based_on_symptoms disease based on symptoms
def predict_disease_based_on_symptoms(symptoms):
    x = mlb.transform([symptoms])

    pred = model.predict(x)[0]
    disease = label_encoder.inverse_transform([pred])[0]

    return disease


if __name__ == "__main__":

    test_inputs = [ 
        [ "cough", "fever"], 
        ["fever", "body pain", "fatigue", "chills"],
        ["headache", "dizziness"],
        ["sneezing", "itching"],
        ["nausea", "bloating"]
    ]
    import random
    rng = random.Random()
    test_input = rng.choice(test_inputs)

    print("\n Input Symptoms:", test_input)
    disease = predict_disease_based_on_symptoms(test_input)
        # import pdb;pdb.set_trace()
    meds = get_extract_medicines(disease)
    print("\n Predicted Disease:", disease)
    print("\n Recommended Medicines:")

    for m in meds:
        print("-", m)
