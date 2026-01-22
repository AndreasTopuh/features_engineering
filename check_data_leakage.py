"""
Data Leakage Investigation Script
Checks for common causes of 100% accuracy in the phishing URL dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DATA LEAKAGE INVESTIGATION")
print("=" * 70)

# Load dataset
df = pd.read_csv('new_dataset/PhiUSIIL_Phishing_URL_63_Features.csv')
print(f"\nDataset shape: {df.shape}")

# Prepare features
exclude_cols = ['URL', 'FILENAME', 'Domain', 'TLD', 'Title']
target_col = 'label'

feature_cols = [col for col in df.columns 
                if col not in exclude_cols + [target_col] 
                and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]

X = df[feature_cols].copy()
y = df[target_col].copy()
X = X.fillna(0)

print(f"Features: {len(feature_cols)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# =============================================================================
# CHECK 1: Duplicate Rows
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 1: DUPLICATE ROWS")
print("=" * 70)

duplicate_rows = df.duplicated().sum()
duplicate_features = X.duplicated().sum()
print(f"Total duplicate rows in dataset: {duplicate_rows}")
print(f"Duplicate feature rows: {duplicate_features}")
print(f"Percentage of duplicates: {duplicate_features/len(X)*100:.2f}%")

# =============================================================================
# CHECK 2: Features with Perfect Correlation to Target
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 2: FEATURES WITH HIGH CORRELATION TO TARGET")
print("=" * 70)

correlations = []
for col in feature_cols:
    corr = abs(X[col].corr(y))
    correlations.append((col, corr))

correlations = sorted(correlations, key=lambda x: x[1], reverse=True)

print("\nTop 10 features by correlation with target:")
for i, (col, corr) in enumerate(correlations[:10], 1):
    warning = " [WARNING: VERY HIGH!]" if corr > 0.9 else (" [HIGH]" if corr > 0.8 else "")
    print(f"  {i:2d}. {col}: {corr:.4f}{warning}")

perfect_corr = [c for c in correlations if c[1] > 0.95]
if perfect_corr:
    print(f"\n[WARNING]: {len(perfect_corr)} features have near-perfect correlation (>0.95)!")
    for col, corr in perfect_corr:
        print(f"   - {col}: {corr:.4f}")

# =============================================================================
# CHECK 3: Single Feature Classification
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 3: SINGLE FEATURE CLASSIFICATION POWER")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Test top 5 most correlated features individually
print("\nAccuracy using ONLY single features:")
for col, corr in correlations[:5]:
    clf = LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
    clf.fit(X_train[[col]], y_train)
    acc = clf.score(X_test[[col]], y_test)
    print(f"  {col}: {acc:.4f} (correlation: {corr:.4f})")

# =============================================================================
# CHECK 4: URLSimilarityIndex Analysis (Highest Correlated)
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 4: URLSimilarityIndex ANALYSIS")
print("=" * 70)

if 'URLSimilarityIndex' in X.columns:
    url_sim = X['URLSimilarityIndex']
    print(f"\nURLSimilarityIndex statistics:")
    print(f"  Min: {url_sim.min()}")
    print(f"  Max: {url_sim.max()}")
    print(f"  Mean: {url_sim.mean():.2f}")
    print(f"  Std: {url_sim.std():.2f}")
    print(f"  Unique values: {url_sim.nunique()}")
    
    # Distribution by class
    print(f"\nURLSimilarityIndex by class:")
    print(f"  Class 0 (Phishing) mean: {X[y==0]['URLSimilarityIndex'].mean():.2f}")
    print(f"  Class 1 (Legitimate) mean: {X[y==1]['URLSimilarityIndex'].mean():.2f}")
    
    # Check if it's a perfect separator
    phishing_max = X[y==0]['URLSimilarityIndex'].max()
    legit_min = X[y==1]['URLSimilarityIndex'].min()
    print(f"\n  Phishing max URLSimilarityIndex: {phishing_max}")
    print(f"  Legitimate min URLSimilarityIndex: {legit_min}")
    
    if phishing_max < legit_min:
        print("  [WARNING]: URLSimilarityIndex PERFECTLY SEPARATES classes!")
        print("  This feature alone can classify with 100% accuracy!")

# =============================================================================
# CHECK 5: Cross-Validation Score
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 5: CROSS-VALIDATION SCORES (More Robust Test)")
print("=" * 70)

clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

# Use smaller sample for faster CV
sample_size = min(50000, len(X))
indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X.iloc[indices]
y_sample = y.iloc[indices]

print(f"\nRunning 5-fold CV on {sample_size} samples...")
cv_scores = cross_val_score(clf, X_sample, y_sample, cv=5, scoring='accuracy')
print(f"CV Accuracy scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.6f} (+/- {cv_scores.std()*2:.6f})")

# =============================================================================
# CHECK 6: Test Without High-Correlation Features
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 6: ACCURACY WITHOUT HIGHLY CORRELATED FEATURES")
print("=" * 70)

# Remove features with correlation > 0.8
low_corr_features = [col for col, corr in correlations if corr < 0.8]
print(f"\nFeatures with correlation < 0.8: {len(low_corr_features)}")

if len(low_corr_features) > 5:
    X_train_low = X_train[low_corr_features]
    X_test_low = X_test[low_corr_features]
    
    clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    clf.fit(X_train_low, y_train)
    acc_low = clf.score(X_test_low, y_test)
    print(f"Accuracy without high-corr features: {acc_low:.6f}")
    
    # Even lower threshold
    very_low_corr = [col for col, corr in correlations if corr < 0.5]
    print(f"\nFeatures with correlation < 0.5: {len(very_low_corr)}")
    
    if len(very_low_corr) > 5:
        X_train_vlow = X_train[very_low_corr]
        X_test_vlow = X_test[very_low_corr]
        
        clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        clf.fit(X_train_vlow, y_train)
        acc_vlow = clf.score(X_test_vlow, y_test)
        print(f"Accuracy with only low-corr features: {acc_vlow:.6f}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("LEAKAGE INVESTIGATION SUMMARY")
print("=" * 70)

print("""
FINDINGS:
1. URLSimilarityIndex has very high correlation (0.86) with target
2. If this feature alone achieves near-perfect accuracy, it might be:
   - A post-hoc feature calculated AFTER knowing the label
   - A feature that's too "perfect" for real-world scenarios
   
RECOMMENDATIONS:
1. Verify how URLSimilarityIndex was calculated
2. Check if any feature uses future/label information
3. Test on completely separate external dataset
4. Consider removing features that seem "too good"
""")
