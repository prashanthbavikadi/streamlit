import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your Excel data
df = pd.read_excel("mycsv.xlsx")

# Replace this with the correct target column from your dataset
target_column = "target"  # 👈 change this to match your data

# Basic preprocessing
X = df.drop(columns=[target_column])
y = df[target_column]

# Handle non-numeric features (optional: customize as needed)
X = pd.get_dummies(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
