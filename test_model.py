import joblib
import pandas as pd

model = joblib.load("trained_model.joblib")

# Each test case: (description, input values, expected result or None if just checking no crash)
test_cases = [
    {"name": "Min Hb boundary", "values": {"Gender": 0, "Age": 25, "Hb": 2.0, "RBC": 4.5, "MCV": 80, "MCH": 27}, "expected": None},
    {"name": "Max Hb boundary", "values": {"Gender": 1, "Age": 30, "Hb": 25.0, "RBC": 5.0, "MCV": 85, "MCH": 29}, "expected": None},
    {"name": "Min Age boundary", "values": {"Gender": 0, "Age": 0, "Hb": 11.0, "RBC": 4.5, "MCV": 70, "MCH": 24}, "expected": None},
    {"name": "Max Age boundary", "values": {"Gender": 1, "Age": 120, "Hb": 11.0, "RBC": 5.0, "MCV": 65, "MCH": 21}, "expected": None},
    {"name": "Clear Normal case", "values": {"Gender": 0, "Age": 30, "Hb": 13.0, "RBC": 4.6, "MCV": 82, "MCH": 28.6}, "expected": "Normal"},
    {"name": "Clear Anemia case", "values": {"Gender": 0, "Age": 27, "Hb": 10.8, "RBC": 4.2, "MCV": 74.5, "MCH": 25.4}, "expected": "Anemia"},
    {"name": "Clear Thalassemia case", "values": {"Gender": 1, "Age": 18, "Hb": 10.6, "RBC": 5.2, "MCV": 61.6, "MCH": 20.0}, "expected": "Thalassemia"},
]

print(f"{'Test':<25} {'Expected':<15} {'Actual':<15} {'Result'}")
print("-" * 70)

passed = 0
for case in test_cases:
    patient = pd.DataFrame([case["values"]])
    prediction = model.predict(patient)[0]

    if case["expected"] is None:
        result = "✅ PASS (no crash)"
        passed += 1
    elif prediction == case["expected"]:
        result = "✅ PASS"
        passed += 1
    else:
        result = "❌ FAIL"

    print(f"{case['name']:<25} {str(case['expected']):<15} {prediction:<15} {result}")

print("-" * 70)
print(f"Total: {passed}/{len(test_cases)} passed")