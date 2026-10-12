import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_excel("data/Anemia_Thalassemia_3Class_Dataset.xlsx")

# Only these columns are real, usable predictors — no leakage
features = ['Gender', 'Age', 'Hb', 'RBC', 'MCV', 'MCH']
target = 'Label'

X = df[features]   # X = input values (what the model looks at)
y = df[target]     # y = the answer (what the model tries to predict)

print(X.head())
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

print("Model trained successfully!")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions, labels=['Normal', 'Anemia', 'Thalassemia'])
print(cm)

joblib.dump(model, "trained_model.joblib")
print("Model saved as trained_model.joblib")