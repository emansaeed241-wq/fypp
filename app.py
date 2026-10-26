import streamlit as st
st.set_page_config(page_title="Anemia & Thalassemia Predictor", page_icon="🩸", layout="centered")

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load our saved, tuned model
model = joblib.load("trained_model.joblib")
features = ['Gender', 'Age', 'Hb', 'RBC', 'MCV', 'MCH']

# This remembers which "page" we're on, across reruns
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ---------------------------------------------------------------
# PAGE 0: WELCOME
# ---------------------------------------------------------------
if st.session_state.page == "welcome":
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center;'>🩸 Anemia & Thalassemia Predictor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-size: 18px;'>"
        "AI-powered screening for blood disorders using CBC test values"
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Let's Get Started ➡", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()

# ---------------------------------------------------------------
# PAGE 1: INPUT FORM
# ---------------------------------------------------------------
elif st.session_state.page == "input":
    st.title("Anemia & Thalassemia Predictor")
    st.write("Enter the patient's blood test values below.")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        hb = st.number_input("Hemoglobin (Hb)", min_value=2.0, max_value=25.0, value=12.0)
    with col2:
        rbc = st.number_input("RBC count", min_value=1.0, max_value=10.0, value=4.8)
        mcv = st.number_input("MCV", min_value=30.0, max_value=150.0, value=85.0)
        mch = st.number_input("MCH", min_value=5.0, max_value=50.0, value=29.0)

    if st.button("Predict", use_container_width=True):
        gender_value = 1 if gender == "Male" else 0

        patient = pd.DataFrame([{
            'Gender': gender_value, 'Age': age, 'Hb': hb,
            'RBC': rbc, 'MCV': mcv, 'MCH': mch
        }])

        prediction = model.predict(patient)[0]
        probabilities = model.predict_proba(patient)[0]
        prob_dict = dict(zip(model.classes_, probabilities))

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient)
        class_index = list(model.classes_).index(prediction)
        class_shap_values = shap_values[0, :, class_index]

        # Save everything we need for the results page
        st.session_state.patient = patient
        st.session_state.prediction = prediction
        st.session_state.prob_dict = prob_dict
        st.session_state.class_shap_values = class_shap_values

        # Switch to the results page
        st.session_state.page = "results"
        st.rerun()

# ---------------------------------------------------------------
# PAGE 2: RESULTS (prediction + chart only)
# ---------------------------------------------------------------
elif st.session_state.page == "results":
    prediction = st.session_state.prediction
    prob_dict = st.session_state.prob_dict

    emoji_map = {"Normal": "🟢", "Anemia": "🟠", "Thalassemia": "🔴"}
    st.subheader(f"{emoji_map.get(prediction, '')} Prediction: {prediction}")

    confidence = prob_dict[prediction]
    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

    st.write("**Class Probabilities**")
    prob_df = pd.DataFrame({
        "Class": list(prob_dict.keys()),
        "Probability": list(prob_dict.values())
    }).set_index("Class")
    st.bar_chart(prob_df)

    st.markdown("---")

    patient = st.session_state.patient
    class_shap_values = st.session_state.class_shap_values

    with st.expander("🔍 Why this prediction?", expanded=True):
        explanation_df = pd.DataFrame({
            'Feature': features,
            'Value': patient.iloc[0].values,
            'Impact': class_shap_values
        }).sort_values('Impact', key=abs, ascending=False)

        top = explanation_df.iloc[0]
        direction = "increased" if top['Impact'] > 0 else "decreased"
        st.write(
            f"The strongest factor was **{top['Feature']}** (value: {top['Value']:.1f}), "
            f"which **{direction}** the likelihood of **{prediction}**."
        )
        st.write("")

        for _, row in explanation_df.iterrows():
            if row['Impact'] > 0:
                st.success(f"🔺 **{row['Feature']}** = {row['Value']:.1f} — pushed **toward** {prediction}")
            else:
                st.error(f"🔻 **{row['Feature']}** = {row['Value']:.1f} — pushed **away from** {prediction}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back to input", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()
    with col2:
        if st.button("🔄 Start over", use_container_width=True):
            st.session_state.page = "welcome"
            st.rerun()
