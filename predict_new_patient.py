import joblib
import pandas as pd

# Load the model we already trained and saved
model = joblib.load("trained_model.joblib")

# A new patient's blood values (make these up, or use real ones you know)
new_patient = pd.DataFrame([{
    'Gender': 0,
    'Age': 25,
    'Hb': 9.5,
    'RBC': 5.4,
    'MCV': 58.0,
    'MCH': 18.0
}])

prediction = model.predict(new_patient)
print("Predicted category:", prediction[0])

probabilities = model.predict_proba(new_patient)
print("Probabilities:", probabilities)