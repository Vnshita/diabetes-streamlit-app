
import joblib
import pandas as pd

def load_models():
    import os
    model_files = [f for f in os.listdir("models") if f.endswith(".pkl") and "encoders" not in f and "scaler" not in f]
    models = {f.replace(".pkl",""): joblib.load(f"models/{f}") for f in model_files}
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return models, scaler, encoders

def encode_inputs(skin, vision, encoders):
    le_skin = encoders["le_skin"]
    le_vision = encoders["le_vision"]
    if skin not in le_skin.classes_:
        skin = "Medium"
    if vision not in le_vision.classes_:
        vision = "Normal"
    return le_skin.transform([skin])[0], le_vision.transform([vision])[0]

def parse_history_flags(history_text):
    text = str(history_text).lower()
    return {
        "Hypertension": int("hypert" in text or "blood pressure" in text),
        "ObesityFlag": int("obes" in text or "obesity" in text or "overweight" in text),
        "FamilyHistory": int("family" in text and ("diabet" in text or "diabetes" in text)) or int("diabet" in text and "family" in text)
    }

def predict_patient(df, models, scaler):
    feature_cols = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
        "BMI","DiabetesPedigreeFunction","Age",
        "PulseRate","SkinColor","Vision",
        "Hypertension","ObesityFlag","FamilyHistory"
    ]
    X = df[feature_cols].copy()
    X_scaled = scaler.transform(X)
    results = []
    for name, mdl in models.items():
        if name.lower()=="logistic_regression":
            y_proba = mdl.predict_proba(X_scaled)[:,1]
            y_pred = mdl.predict(X_scaled)
        else:
            y_proba = mdl.predict_proba(X)[:,1]
            y_pred = mdl.predict(X)
        results.append((name, y_pred, y_proba))
    return results
