import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# TITLE
# -----------------------------------
st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below to predict heart disease risk.")

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = joblib.load("model.pkl")

# -----------------------------------
# TRAINING COLUMN ORDER (IMPORTANT)
# -----------------------------------
model_columns = [
'slope_of_peak_exercise_st_segment',
'thal',
'resting_blood_pressure',
'chest_pain_type',
'num_major_vessels',
'fasting_blood_sugar_gt_120_mg_per_dl',
'resting_ekg_results',
'serum_cholesterol_mg_per_dl',
'oldpeak_eq_st_depression',
'sex',
'age',
'max_heart_rate_achieved',
'exercise_induced_angina'
]

# -----------------------------------
# USER INPUT
# -----------------------------------
st.subheader("Patient Details")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    [0,1,2,3]
)

resting_blood_pressure = st.number_input(
    "Resting Blood Pressure",
    80, 200, 120
)

serum_cholesterol_mg_per_dl = st.number_input(
    "Serum Cholesterol (mg/dl)",
    100, 600, 200
)

fasting_blood_sugar_gt_120_mg_per_dl = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0,1]
)

resting_ekg_results = st.selectbox(
    "Resting ECG Results",
    [0,1,2]
)

max_heart_rate_achieved = st.number_input(
    "Max Heart Rate Achieved",
    60, 220, 150
)

exercise_induced_angina = st.selectbox(
    "Exercise Induced Angina",
    [0,1]
)

oldpeak_eq_st_depression = st.number_input(
    "ST Depression (Oldpeak)",
    0.0, 10.0, 1.0
)

slope_of_peak_exercise_st_segment = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    [0,1,2]
)

num_major_vessels = st.selectbox(
    "Number of Major Vessels",
    [0,1,2,3]
)

thal = st.selectbox(
    "Thal",
    [0,1,2,3]
)

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("Predict Heart Disease"):

    input_data = {
        'slope_of_peak_exercise_st_segment': slope_of_peak_exercise_st_segment,
        'thal': thal,
        'resting_blood_pressure': resting_blood_pressure,
        'chest_pain_type': chest_pain_type,
        'num_major_vessels': num_major_vessels,
        'fasting_blood_sugar_gt_120_mg_per_dl': fasting_blood_sugar_gt_120_mg_per_dl,
        'resting_ekg_results': resting_ekg_results,
        'serum_cholesterol_mg_per_dl': serum_cholesterol_mg_per_dl,
        'oldpeak_eq_st_depression': oldpeak_eq_st_depression,
        'sex': sex,
        'age': age,
        'max_heart_rate_achieved': max_heart_rate_achieved,
        'exercise_induced_angina': exercise_induced_angina
    }

    input_df = pd.DataFrame([input_data])

    # SAME ORDER AS TRAINING
    input_df = input_df[model_columns]

    # IMPORTANT FIX (prevents isnan error)
    input_df = input_df.astype(float)

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Risk Detected")
    else:
        st.success("✅ No Heart Disease Detected")