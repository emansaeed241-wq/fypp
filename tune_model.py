import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_excel("data/Anemia_Thalassemia_3Class_Dataset.xlsx")

features = ['Gender', 'Age', 'Hb', 'RBC', 'MCV', 'MCH']
target = 'Label'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# These are the combinations GridSearch will try
param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [None, 8, 12],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    scoring='f1_macro',
    cv=5
)

grid_search.fit(X_train, y_train)

print("Best settings found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

import joblib

# Use the best settings GridSearch found to build our final model
best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)
print(classification_report(y_test, predictions))

joblib.dump(best_model, "trained_model.joblib")
print("Final tuned model saved!")