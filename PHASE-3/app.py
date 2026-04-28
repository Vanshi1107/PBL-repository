import streamlit as st
import pandas as pd
import pickle

# Load saved objects
model = pickle.load(open("model/logistic_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
selector = pickle.load(open("model/selector.pkl", "rb"))
features = pickle.load(open("model/all_features.pkl", "rb"))

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details below:")

# ---- INPUTS ---- #

age = st.number_input("Age", 1, 120, 40)

sex_label = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_label == "Male" else 0

cp_label = st.selectbox("Chest Pain Type", [
    "Typical Angina (0)",
    "Atypical Angina (1)",
    "Non-anginal Pain (2)",
    "Asymptomatic (3)"
])
cp = int(cp_label[-2])

trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

chol = st.number_input("Cholesterol", 100, 600, 200)

fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [
    "No (0)",
    "Yes (1)"
])
fbs = int(fbs_label[-2])

restecg_label = st.selectbox("Resting ECG", [
    "Normal (0)",
    "ST-T Abnormality (1)",
    "Left Ventricular Hypertrophy (2)"
])
restecg = int(restecg_label[-2])

thalach = st.number_input("Max Heart Rate", 60, 220, 150)

exang_label = st.selectbox("Exercise Induced Angina", [
    "No (0)",
    "Yes (1)"
])
exang = int(exang_label[-2])

oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, 1.0)

slope_label = st.selectbox("Slope of Peak Exercise ST Segment", [
    "Upsloping (0)",
    "Flat (1)",
    "Downsloping (2)"
])
slope = int(slope_label[-2])

ca_label = st.selectbox("Number of Major Vessels", ["0", "1", "2", "3"])
ca = int(ca_label)

thal_label = st.selectbox("Thalassemia", [
    "Normal (1)",
    "Fixed Defect (2)",
    "Reversible Defect (3)"
])
thal = int(thal_label[-2])

# ---- PREDICT BUTTON ---- #

if st.button("Predict"):

    # Create full feature set
    input_data = dict.fromkeys(features, 0)

    # Numerical features
    input_data["Age"] = age
    input_data["RestingBP"] = trestbps
    input_data["Cholesterol"] = chol
    input_data["MaxHR"] = thalach
    input_data["Oldpeak"] = oldpeak
    input_data["FastingBS"] = fbs
    input_data["CA"] = ca
    input_data["Thal"] = thal

    # ---- One-hot encoding ---- #

    # Sex
    if sex == 1:
        input_data["Sex_M"] = 1

    # Chest Pain
    cp_map = {
        0: "ChestPainType_TA",
        1: "ChestPainType_ATA",
        2: "ChestPainType_NAP",
        3: "ChestPainType_ASY"
    }
    input_data[cp_map[cp]] = 1

    # Exercise Angina
    if exang == 1:
        input_data["ExerciseAngina_Y"] = 1

    # ST Slope
    slope_map = {
        0: "ST_Slope_Up",
        1: "ST_Slope_Flat",
        2: "ST_Slope_Down"
    }
    input_data[slope_map[slope]] = 1

    # Rest ECG
    restecg_map = {
        0: "RestingECG_Normal",
        1: "RestingECG_ST",
        2: "RestingECG_LVH"
    }
    if restecg_map[restecg] in input_data:
        input_data[restecg_map[restecg]] = 1

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    try:
        df = df[features]

        scaled = scaler.transform(df)
        selected = selector.transform(scaled)

        pred = model.predict(selected)[0]
        prob = model.predict_proba(selected)[0][1]

        st.write(f"Prediction: {pred}")
        st.write(f"Probability of Disease: {prob:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")