import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("data/Anemia_Thalassemia_3Class_Dataset.xlsx")

df['Label'].value_counts().plot(kind='bar')
plt.title("Patients in each category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()


sns.boxplot(data=df, x='Label', y='Hb')
plt.title("Hb Distribution by Category")
plt.show()


sns.boxplot(data=df, x='Label', y='MCV')
plt.title("MCV Distribution by Category")
plt.show()

sns.boxplot(data=df, x='Label', y='RBC')
plt.title("RBC Distribution by Category")
plt.show()

numeric_df = df[['Gender', 'Age', 'Hb', 'RBC', 'MCV', 'MCH']]
print(numeric_df.corr())