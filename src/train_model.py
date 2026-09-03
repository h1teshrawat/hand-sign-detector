import pandas as pd
import numpy as np
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neural_network   import MLPClassifier
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import classification_report, accuracy_score
import pickle

# Load dataset
df = pd.read_csv("data/hand_signs.csv", header=None)

# First column = label, rest = 63 landmark features
X = df.iloc[:, 1:].values   # shape: (N, 63)
y = df.iloc[:, 0].values    # shape: (N,)

# Encode labels A-Z → 0-25
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Classes          : {list(le.classes_)}")

# --- Option 1: Random Forest (fast, good baseline) ---
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("\n=== Random Forest ===")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# --- Option 2: MLP (slightly better for complex signs) ---
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    max_iter=500,
    random_state=42
)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
print("\n=== MLP ===")
print(f"Accuracy: {accuracy_score(y_test, mlp_pred):.3f}")
print(classification_report(y_test, mlp_pred, target_names=le.classes_))

# Save the better model + encoder
with open("model/sign_model.pkl", "wb") as f:
    pickle.dump(rf, f)      # swap to mlp if MLP was better
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nModel saved to model/sign_model.pkl")