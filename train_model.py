import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("preprocessed_o.csv")
df = df.dropna()

df = df[df['smoking_status'] != 'Unknown']


# Encode categorical columns
le = LabelEncoder()
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df[col] = le.fit_transform(df[col])

# Split features and label
X = df.drop(['Patient_id', 'stroke'], axis=1)
y = df['stroke']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and column names
joblib.dump(model, "stroke_model.pkl")
joblib.dump(X.columns, "model_columns.pkl")

print("✅ Model and column names saved successfully.")
