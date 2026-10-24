import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

df = pd.read_excel("data/Anemia_Thalassemia_3Class_Dataset.xlsx")

# NEW FEATURE: Mentzer Index — a real clinical formula doctors use to
# distinguish Thalassemia from Iron-Deficiency Anemia
df['Mentzer_Index'] = df['MCV'] / df['RBC']

print(df[['MCV', 'RBC', 'Mentzer_Index', 'Label']].head(10))