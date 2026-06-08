import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data
from train_model import train_model
from fairness_metrics import calculate_fairness
from mitigation import mitigate_bias


# STEP 1 — Load Dataset from OpenML
data = fetch_openml("credit-g", version=1, as_frame=True)

df = data.frame

print("\nDataset Columns:")
print(df.columns)


# STEP 2 — Preprocess Dataset
df = preprocess_data(df)


# STEP 3 — Define Features and Target
X = df.drop("class", axis=1)

y = df["class"]


# STEP 4 — Define Sensitive Feature
# STEP 4 — Create Age Groups

df["age_group"] = (df["age"] >= 25).astype(int)

# 0 = Under 25
# 1 = 25 and Over

# STEP 5 — Define Features and Target

X = df.drop(["class", "age_group"], axis=1)

y = df["class"]

sensitive_feature = df["age_group"]


# STEP 5 — Train Test Split
X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
    X,
    y,
    sensitive_feature,
    test_size=0.3,
    random_state=42
)


# STEP 6 — Train Baseline Model
model = train_model(X_train, y_train)


# STEP 7 — Baseline Predictions
y_pred = model.predict(X_test)


# STEP 8 — Baseline Accuracy
baseline_accuracy = accuracy_score(y_test, y_pred)

print("\nBaseline Accuracy:")
print(baseline_accuracy)


# STEP 9 — Baseline Fairness
baseline_fairness = calculate_fairness(
    y_test,
    y_pred,
    sf_test
)

print("\nBaseline Demographic Parity Difference:")
print(baseline_fairness)


# STEP 10 — Apply Mitigation
mitigator = mitigate_bias(
    X_train,
    y_train,
    sf_train
)


# STEP 11 — Mitigated Predictions
y_pred_mitigated = mitigator.predict(X_test)


# STEP 12 — Mitigated Accuracy
mitigated_accuracy = accuracy_score(
    y_test,
    y_pred_mitigated
)

print("\nMitigated Accuracy:")
print(mitigated_accuracy)


# STEP 13 — Mitigated Fairness
mitigated_fairness = calculate_fairness(
    y_test,
    y_pred_mitigated,
    sf_test
)

print("\nMitigated Demographic Parity Difference:")
print(mitigated_fairness)


# STEP 14 — Final Comparison Table
results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Demographic Parity Difference"
    ],
    "Before Mitigation": [
        baseline_accuracy,
        baseline_fairness
    ],
    "After Mitigation": [
        mitigated_accuracy,
        mitigated_fairness
    ]
})

print("\nFinal Results Comparison:")
print(results)


# STEP 15 — Save Results
results.to_csv("../fairness_results.csv", index=False)

print("\nResults saved successfully.")