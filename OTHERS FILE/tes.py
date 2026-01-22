
import pandas as pd

files = [
    ('63_Features', 'new_dataset/PhiUSIIL_Phishing_URL_63_Features.csv'),
    ('59_Features_VALIDATED', 'new_dataset/PhiUSIIL_Phishing_URL_59_Features_VALIDATED - Copy.csv'),
    ('33_Features_SELECTED', 'new_dataset/PhiUSIIL_Phishing_URL_33_Features_SELECTED.csv'),
    ('OLD_Dataset', 'old-data/5PhiUSIIL_Phishing_URL_Dataset_OLD.csv')
]

all_columns = {}
for name, path in files:
    try:
        df = pd.read_csv(path, nrows=1)
        all_columns[name] = set(df.columns.tolist())
        print(f'\n=== {name} ({len(df.columns)} columns) ===')
        for i, col in enumerate(df.columns, 1):
            print(f'  {i:2d}. {col}')
    except Exception as e:
        print(f'Error reading {name}: {e}')

# Compare
print('\n' + '='*80)
print('FEATURE COMPARISON')
print('='*80)

# Common in all
common_all = all_columns['63_Features'] & all_columns['59_Features_VALIDATED'] & all_columns['33_Features_SELECTED'] & all_columns['OLD_Dataset']
print(f'\nCommon in ALL 4 datasets: {len(common_all)} features')

# Only in 63
only_63 = all_columns['63_Features'] - all_columns['OLD_Dataset']
print(f'\nNEW features in 63_Features (not in OLD): {len(only_63)}')
for f in sorted(only_63):
    print(f'  + {f}')

# Only in OLD
only_old = all_columns['OLD_Dataset'] - all_columns['63_Features']
print(f'\nFeatures REMOVED from OLD (not in 63_Features): {len(only_old)}')
for f in sorted(only_old):
    print(f'  - {f}')

# 33 Selected features
print(f'\n33_Features_SELECTED contains: {len(all_columns["33_Features_SELECTED"])} features')
