import streamlit as st
st.set_page_config(page_title="Anemia & Thalassemia Predictor", page_icon="🩸", layout="centered")

st.sidebar.header("About")
st.sidebar.write(
    "This tool predicts Normal / Anemia / Thalassemia from CBC blood values "
    "using a trained Random Forest model, with SHAP-based explanations."
)
st.sidebar.warning(
    "⚠️ This is a screening aid for a Final Year Project, NOT a medical "
    "diagnosis. Always consult a qualified doctor."
)
import joblib
import pandas as pd
import shap

# Load our saved, tuned model
model = joblib.load("trained_model.joblib")

st.title("Anemia & Thalassemia Predictor")
st.write("Enter the patient's blood test values below.")

# Input fields — user yahan values daalega
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    hb = st.number_input("Hemoglobin (Hb)", min_value=2.0, max_value=25.0, value=12.0)

with col2:
    rbc = st.number_input("RBC count", min_value=1.0, max_value=10.0, value=4.8)
    mcv = st.number_input("MCV", min_value=30.0, max_value=150.0, value=85.0)
    mch = st.number_input("MCH", min_value=5.0, max_value=50.0, value=29.0)

features = ['Gender', 'Age', 'Hb', 'RBC', 'MCV', 'MCH']

if st.button("Predict"):
    gender_value = 1 if gender == "Male" else 0

    patient = pd.DataFrame([{
        'Gender': gender_value,
        'Age': age,
        'Hb': hb,
        'RBC': rbc,
        'MCV': mcv,
        'MCH': mch
    }])

    prediction = model.predict(patient)[0]
    st.markdown("---")

    emoji_map = {"Normal": "🟢", "Anemia": "🟠", "Thalassemia": "🔴"}
    st.subheader(f"{emoji_map.get(prediction, '')} Prediction: {prediction}")

    probabilities = model.predict_proba(patient)[0]
    prob_dict = dict(zip(model.classes_, probabilities))
    confidence = prob_dict[prediction]
    st.progress(confidence, text=f"Confidence: {confidence:.1%}")
    
    st.write("**Class Probabilities**")
    prob_df = pd.DataFrame({
        "Class": list(prob_dict.keys()),
        "Probability": list(prob_dict.values())
    }).set_index("Class")
    st.bar_chart(prob_df)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient)

    class_index = list(model.classes_).index(prediction)
    class_shap_values = shap_values[0, :, class_index]

    st.write("### Why this prediction?")
    explanation_df = pd.DataFrame({
        'Feature': features,
        'Value': patient.iloc[0].values,
        'Impact': class_shap_values
    }).sort_values('Impact', key=abs, ascending=False)

    st.dataframe(explanation_df)