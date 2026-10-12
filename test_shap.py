import joblib
import pandas as pd
import shap

model = joblib.load("trained_model.joblib")

# Same patient as before (Thalassemia-like)
patient = pd.DataFrame([{
    'Gender': 0,
    'Age': 25,
    'Hb': 9.5,
    'RBC': 5.4,
    'MCV': 58.0,
    'MCH': 18.0
}])

# TreeExplainer works fast and exact for tree-based models like Random Forest
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(patient)

print("Type:", type(shap_values))
print("Model classes order:", model.classes_)

import numpy as np
print("Shape:", np.array(shap_values).shape)

# Find which position "Thalassemia" is in the classes list
predicted_class = model.predict(patient)[0]
class_index = list(model.classes_).index(predicted_class)
print("Predicted class:", predicted_class, "| index:", class_index)

# Get SHAP values just for the predicted class
class_shap_values = shap_values[0, :, class_index]

features = ['Gender', 'Age', 'Hb', 'RBC', 'MCV', 'MCH']
for fname, value in zip(features, class_shap_values):
    print(fname, ":", round(value, 4))