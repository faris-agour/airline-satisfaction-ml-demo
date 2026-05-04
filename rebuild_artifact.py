import numpy as np
import pandas as pd
import kagglehub
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

FEATURE_ORDER = [
    "Gender",
    "Customer Type",
    "Age",
    "Type of Travel",
    "Class",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

NUMERIC_FEATURES = [f for f in FEATURE_ORDER if f not in CATEGORICAL_FEATURES]

FEATURE_DESCRIPTIONS = {
    "Gender": "Passenger gender.",
    "Customer Type": "Loyal customer status.",
    "Age": "Passenger age in years.",
    "Type of Travel": "Trip purpose: business or personal.",
    "Class": "Travel class (Business, Eco, Eco Plus).",
    "Flight Distance": "Travel distance in miles.",
    "Inflight wifi service": "Rating from 0 to 5.",
    "Departure/Arrival time convenient": "Convenience rating from 0 to 5.",
    "Ease of Online booking": "Online booking experience rating from 0 to 5.",
    "Gate location": "Gate location convenience rating from 0 to 5.",
    "Food and drink": "Food and drink quality rating from 0 to 5.",
    "Online boarding": "Online boarding experience rating from 0 to 5.",
    "Seat comfort": "Seat comfort rating from 0 to 5.",
    "Inflight entertainment": "Inflight entertainment rating from 0 to 5.",
    "On-board service": "On-board service quality rating from 0 to 5.",
    "Leg room service": "Leg room comfort rating from 0 to 5.",
    "Baggage handling": "Baggage handling service rating from 0 to 5.",
    "Checkin service": "Check-in service rating from 0 to 5.",
    "Inflight service": "Inflight service quality rating from 0 to 5.",
    "Cleanliness": "Cleanliness rating from 0 to 5.",
}

path = kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")
df = pd.read_csv(path + "/train.csv")
df = df.drop(columns=["Unnamed: 0", "id"]).copy()

for col in FEATURE_ORDER:
    if col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode().iloc[0]).astype(str)
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

y = (df["satisfaction"].astype(str).str.strip().str.lower() == "satisfied").astype(int)
X_raw = df[FEATURE_ORDER].copy()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

category_to_code = {}
categorical_options = {}

for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    X_train_raw[col] = le.fit_transform(X_train_raw[col])
    mapping = {label: int(code) for code, label in enumerate(le.classes_)}
    X_test_raw[col] = X_test_raw[col].map(mapping).fillna(-1).astype(int)

    category_to_code[col] = mapping
    categorical_options[col] = list(le.classes_)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw[FEATURE_ORDER])
X_test = scaler.transform(X_test_raw[FEATURE_ORDER])

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)

feature_bounds = {}
feature_defaults = {}

for col in NUMERIC_FEATURES:
    min_v = float(df[col].min())
    max_v = float(df[col].max())
    default_v = float(df[col].median())

    is_int = pd.api.types.is_integer_dtype(df[col]) or np.allclose(df[col] % 1, 0)

    feature_bounds[col] = {
        "min": int(min_v) if is_int else min_v,
        "max": int(max_v) if is_int else max_v,
        "step": 1 if is_int else 0.1,
        "is_int": bool(is_int),
    }
    feature_defaults[col] = int(default_v) if is_int else default_v


def build_demo_case(dataframe):
    case = {}
    for col in FEATURE_ORDER:
        if col in CATEGORICAL_FEATURES:
            case[col] = str(dataframe[col].mode().iloc[0])
        else:
            value = float(dataframe[col].median())
            if col in feature_bounds and feature_bounds[col]["is_int"]:
                value = int(round(value))
            case[col] = value
    return case


dissatisfied_df = df[df["satisfaction"].str.lower().eq("neutral or dissatisfied")]
satisfied_df = df[df["satisfaction"].str.lower().eq("satisfied")]

demo_test_cases = {
    "Dissatisfied": build_demo_case(dissatisfied_df),
    "Satisfied": build_demo_case(satisfied_df),
}

artifact = {
    "model_name": "Random Forest",
    "model": model,
    "scaler": scaler,
    "feature_order": FEATURE_ORDER,
    "categorical_features": CATEGORICAL_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "feature_descriptions": FEATURE_DESCRIPTIONS,
    "category_to_code": category_to_code,
    "categorical_options": categorical_options,
    "feature_bounds": feature_bounds,
    "feature_defaults": feature_defaults,
    "label_map": {0: "Neutral or Dissatisfied", 1: "Satisfied"},
    "metrics": {"accuracy": float(acc), "f1": float(f1)},
    "demo_test_cases": demo_test_cases,
}

output_path = Path(__file__).with_name("best_model.joblib")
joblib.dump(artifact, output_path, compress=3)

print("Saved:", output_path.resolve())
print("Accuracy:", round(acc, 4))
print("F1:", round(f1, 4))
print("Features:", FEATURE_ORDER)
print("Demo Dissatisfied case:", demo_test_cases["Dissatisfied"])
print("Demo Satisfied case:", demo_test_cases["Satisfied"])
