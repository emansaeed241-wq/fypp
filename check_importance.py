import joblib, pandas as pd
model = joblib.load("trained_model.joblib")
print(pd.Series(model.feature_importances_, index=['Gender','Age','Hb','RBC','MCV','MCH']).sort_values(ascending=False))