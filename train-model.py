import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import kagglehub
import os

# Step 1: Download and load the dataset
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
csv_file = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(csv_file)

# Step 2: Clean and preprocess the dataset
df.replace(" ", pd.NA, inplace=True)
df.dropna(inplace=True)  # Handle missing TotalCharges
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Drop non-informative columns
df.drop(columns=["customerID"], inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include="object").columns.drop("Churn")
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Scale numeric columns
scaler = StandardScaler()
df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(
    df[["tenure", "MonthlyCharges", "TotalCharges"]]
)

# Step 3: Train-test split and model training
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 4: Save the model and scaler
joblib.dump(model, "logreg_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved as logreg_model.pkl and scaler.pkl")
