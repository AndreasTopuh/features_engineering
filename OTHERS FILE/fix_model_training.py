"""
FIX MODEL TRAINING - Hyperparameters dan Data Leakage Investigation
Berdasarkan Paper Prasad (Computers & Security 136, 2024)

MASALAH: Akurasi 100% mengindikasikan kemungkinan data leakage
SOLUSI: Script ini akan menginvestigasi dan memperbaiki model training
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

print("=" * 80)
print("INVESTIGATION: MENGAPA AKURASI 100%?")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA DAN CEK DATA LEAKAGE
# ============================================================================
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('new_dataset/PhiUSIIL_Phishing_URL_63_Features.csv')
print(f"Dataset shape: {df.shape}")

# Kolom yang harus DIEXCLUDE (bukan fitur, tapi identifiers)
exclude_cols = ['URL', 'FILENAME', 'Domain', 'TLD', 'Title', 'Unnamed: 55']
target_col = 'label'

# Cek apakah ada fitur FILENAME yang bisa menyebabkan leakage
print("\n[INVESTIGATION] Cek apakah FILENAME mengandung pola label...")
if 'FILENAME' in df.columns:
    # Cek unik FILENAME
    print(f"  - Unique FILENAME count: {df['FILENAME'].nunique()}")
    print(f"  - Total rows: {len(df)}")
    if df['FILENAME'].nunique() == len(df):
        print("  ✓ Setiap row memiliki FILENAME unik (tidak menyebabkan leakage)")
    else:
        print("  ⚠ Ada FILENAME duplikat!")

# ============================================================================
# STEP 2: CEK FITUR YANG TERLALU BERKORELASI DENGAN LABEL
# ============================================================================
print("\n[STEP 2] Checking feature correlation with label...")

# Ambil hanya kolom numerik
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in [target_col] + exclude_cols]

print(f"  - Total numeric features: {len(feature_cols)}")

# Hitung korelasi dengan label
correlations = {}
for col in feature_cols:
    correlations[col] = abs(df[col].corr(df[target_col]))

# Sort by correlation
sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

print("\n  TOP 10 FITUR YANG PALING BERKORELASI DENGAN LABEL:")
print("  " + "-" * 50)
high_corr_features = []
for i, (feat, corr) in enumerate(sorted_corr[:10], 1):
    warning = " ⚠ SUSPICIOUS!" if corr > 0.95 else ""
    print(f"  {i:2d}. {feat:35s} : {corr:.6f}{warning}")
    if corr > 0.95:
        high_corr_features.append(feat)

if high_corr_features:
    print(f"\n  ⚠ WARNING: {len(high_corr_features)} fitur memiliki korelasi > 0.95!")
    print(f"  Fitur tersebut: {high_corr_features}")
    print("  Ini bisa menyebabkan data leakage!")

# ============================================================================
# STEP 3: PREPARE DATA DENGAN BENAR
# ============================================================================
print("\n[STEP 3] Preparing data correctly...")

# Remove high correlation features if any
feature_cols_clean = [col for col in feature_cols if col not in high_corr_features]
print(f"  - Original features: {len(feature_cols)}")
print(f"  - After removing suspicious features: {len(feature_cols_clean)}")

X = df[feature_cols_clean].copy()
y = df[target_col].copy()

# Handle missing values
X = X.fillna(0)

print(f"  - Feature matrix shape: {X.shape}")
print(f"  - Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# STEP 4: SPLIT DATA (80% TRAIN, 20% TEST)
# ============================================================================
print("\n[STEP 4] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  - Training set: {X_train.shape[0]} samples")
print(f"  - Test set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 5: TRAIN MODELS DENGAN DEFAULT HYPERPARAMETERS (SESUAI PAPER)
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING MODELS (Hyperparameters sesuai dengan Paper Prasad)")
print("=" * 80)

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Train model and return metrics with 7 decimal precision"""
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics with high precision
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    return {
        'Model': model_name,
        'Accuracy': round(accuracy, 7),
        'Precision': round(precision, 7),
        'Recall': round(recall, 7),
        'F1-Score': round(f1, 7),
        'MCC': round(mcc, 7),
        'Training Time (s)': round(training_time, 2)
    }

# Define models with DEFAULT hyperparameters (seperti yang kemungkinan digunakan di paper)
models = {
    'LightGBM': LGBMClassifier(
        n_estimators=100,      # default
        learning_rate=0.1,     # default
        max_depth=-1,          # unlimited (default)
        num_leaves=31,         # default
        random_state=42,
        verbose=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,      # default
        learning_rate=0.3,     # default
        max_depth=6,           # default
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    ),
    'CatBoost': CatBoostClassifier(
        iterations=100,        # similar to n_estimators
        learning_rate=0.03,    # CatBoost auto
        depth=6,               # default
        random_state=42,
        verbose=0
    )
}

results = []
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    result = train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name)
    results.append(result)
    print(f"  Accuracy: {result['Accuracy']:.7f}")
    print(f"  F1-Score: {result['F1-Score']:.7f}")
    print(f"  MCC: {result['MCC']:.7f}")
    print(f"  Time: {result['Training Time (s)']}s")

# ============================================================================
# STEP 6: DISPLAY RESULTS TABLE
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS TABLE (7 Decimal Precision)")
print("=" * 80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ============================================================================
# STEP 7: COMPARE WITH PAPER RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON WITH PAPER (Prasad, Computers & Security 136, 2024)")
print("=" * 80)

paper_results = {
    'LightGBM': {'Accuracy': 0.9999, 'Precision': 0.99991, 'Recall': 0.99993, 'F1-Score': 0.99992, 'MCC': 0.99981},
    'XGBoost': {'Accuracy': 0.99993, 'Precision': 0.99993, 'Recall': 0.99994, 'F1-Score': 0.99994, 'MCC': 0.99985},
    'CatBoost': {'Accuracy': 0.99987, 'Precision': 0.99981, 'Recall': 0.99996, 'F1-Score': 0.99989, 'MCC': 0.99974}
}

print("\n{:<12} {:>15} {:>15}".format("Model", "Your Accuracy", "Paper Accuracy"))
print("-" * 45)
for result in results:
    model = result['Model']
    your_acc = result['Accuracy']
    paper_acc = paper_results.get(model, {}).get('Accuracy', 'N/A')
    diff = your_acc - paper_acc if isinstance(paper_acc, float) else 'N/A'
    diff_str = f"{diff:+.7f}" if isinstance(diff, float) else diff
    print(f"{model:<12} {your_acc:>15.7f} {paper_acc:>15.5f} (diff: {diff_str})")

# ============================================================================
# STEP 8: CONFUSION MATRIX
# ============================================================================
print("\n" + "=" * 80)
print("CONFUSION MATRICES")
print("=" * 80)

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name}:")
    print(f"  TN: {cm[0][0]:>6}  FP: {cm[0][1]:>6}")
    print(f"  FN: {cm[1][0]:>6}  TP: {cm[1][1]:>6}")

# Save results
results_df.to_csv('fixed_model_results.csv', index=False)
print("\n\n✓ Results saved to: fixed_model_results.csv")

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
