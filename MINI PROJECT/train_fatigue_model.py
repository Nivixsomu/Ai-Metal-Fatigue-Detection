import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib

DATASET_PATH = "features_with_gt_labels.csv"  # change if needed

# Load data or make dummy if missing
if os.path.exists(DATASET_PATH):
    print(f"✅ Found dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df_labeled = df[df["gt_label"].notna()].copy()
    feature_cols = [
        "n_events","n_chars","backspace_count","duration_sec","wpm",
        "avg_hold_sec","std_hold_sec","avg_flight_sec","std_flight_sec","max_idle_sec"
    ]
    X = df_labeled[feature_cols]
    y = df_labeled["gt_label"].astype(int)

    # ⚖️ Balance dataset
    class_counts = y.value_counts()
    print(f"Class distribution before balancing:\n{class_counts}\n")

    majority_class = y.value_counts().idxmax()
    minority_class = y.value_counts().idxmin()

    df_majority = df_labeled[df_labeled.gt_label == majority_class]
    df_minority = df_labeled[df_labeled.gt_label == minority_class]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,                     # sample with replacement
        n_samples=len(df_majority),       # match majority size
        random_state=42
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    X = df_balanced[feature_cols]
    y = df_balanced["gt_label"].astype(int)

    print(f"✅ Balanced dataset shape: {X.shape}, class counts:\n{y.value_counts()}\n")

else:
    print("⚠️ Dataset not found, generating dummy training data...")
    np.random.seed(42)
    X = np.random.rand(500, 10) * 10
    y = np.random.randint(0, 2, 500)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Pipeline
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
pipe.fit(X_train, y_train)

print("\nClassification Report:\n")
print(classification_report(y_test, pipe.predict(X_test)))

# Save model
MODEL_PATH = "fatigue_demo_model.pkl"
joblib.dump(pipe, MODEL_PATH)
print(f"\n✅ Model saved as: {MODEL_PATH}")
