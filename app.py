
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_models, encode_inputs, parse_history_flags, predict_patient

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("💉 Diabetes Prediction App")

models, scaler, encoders = load_models()

st.sidebar.header("Enter Patient Details")
num_patients = st.sidebar.number_input("Number of Patients", min_value=1, max_value=10, value=1)
patient_data = []
for i in range(num_patients):
    st.sidebar.subheader(f"Patient {i+1}")
    name = st.sidebar.text_input(f"Name {i+1}", f"Patient {i+1}")
    preg = st.sidebar.number_input(f"Pregnancies {i+1}", 0, 20, 2)
    glu = st.sidebar.number_input(f"Glucose {i+1}", 50, 300, 120)
    bp = st.sidebar.number_input(f"BloodPressure {i+1}", 50, 200, 80)
    st_skin = st.sidebar.number_input(f"SkinThickness {i+1}", 5, 100, 25)
    ins = st.sidebar.number_input(f"Insulin {i+1}", 0, 1000, 85)
    bmi = st.sidebar.number_input(f"BMI {i+1}", 10.0, 60.0, 28.5)
    dpf = st.sidebar.number_input(f"DPF {i+1}", 0.0, 2.5, 0.5)
    age = st.sidebar.number_input(f"Age {i+1}", 1, 120, 45)
    pulse = st.sidebar.number_input(f"PulseRate {i+1}", 40, 200, 78)
    skin_color = st.sidebar.selectbox(f"Skin Color {i+1}", ["Fair","Medium","Dark"])
    vision = st.sidebar.selectbox(f"Vision {i+1}", ["Normal","Blurred","Impaired"])
    history = st.sidebar.text_input(f"Medical History {i+1}", "Hypertension, family_diabetes")

    skin_enc, vision_enc = encode_inputs(skin_color, vision, encoders)
    flags = parse_history_flags(history)
    patient_data.append({
        "Name": name,
        "Pregnancies": preg,
        "Glucose": glu,
        "BloodPressure": bp,
        "SkinThickness": st_skin,
        "Insulin": ins,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "PulseRate": pulse,
        "SkinColor": skin_enc,
        "Vision": vision_enc,
        "Hypertension": flags["Hypertension"],
        "ObesityFlag": flags["ObesityFlag"],
        "FamilyHistory": flags["FamilyHistory"]
    })

patients_df = pd.DataFrame(patient_data)

if st.button("Predict All Patients"):
    all_results = []
    for idx, row in patients_df.iterrows():
        df_row = pd.DataFrame([row.drop("Name")])
        preds = predict_patient(df_row, models, scaler)
        st.subheader(f"Patient {idx+1}: {row['Name']}")
        patient_results = []
        for model_name, y_pred, y_proba in preds:
            st.write(f"**{model_name}**: {'Diabetic' if y_pred[0]==1 else 'Non-Diabetic'} (Prob: {y_proba[0]:.2f})")
            plt.figure(figsize=(4,2))
            sns.barplot(x=[model_name], y=[y_proba[0]])
            plt.ylim(0,1)
            plt.title(f"{model_name} - Prob")
            st.pyplot(plt)
            plt.clf()
            patient_results.append({"Model": model_name, "Prediction": "Diabetic" if y_pred[0]==1 else "Non-Diabetic", "Prob": float(y_proba[0])})
        all_results.append({"Patient": row["Name"], "Results": patient_results})

    csv_rows = []
    for patient in all_results:
        for r in patient["Results"]:
            csv_rows.append({"Patient": patient["Patient"], **r})
    csv_df = pd.DataFrame(csv_rows)
    st.download_button("Download Predictions CSV", csv_df.to_csv(index=False), "predictions.csv", "text/csv")
