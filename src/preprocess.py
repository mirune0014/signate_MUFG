import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    data_dir = Path('data/input')
    output_dir = Path('data/output')
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')

    target_col = 'LoanStatus'
    id_col = 'id'

    # check missing values
    if train.isnull().values.any() or test.isnull().values.any():
        raise ValueError('Missing values found in datasets')

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.copy()

    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols + [id_col]]

    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    X_train_proc = preprocess.fit_transform(X_train.drop(columns=[id_col]))
    X_test_proc = preprocess.transform(X_test.drop(columns=[id_col]))

    np.savez_compressed(output_dir / 'train_preprocessed.npz', X=X_train_proc, y=y_train.values)
    np.savez_compressed(output_dir / 'test_preprocessed.npz', X=X_test_proc)

    info = {
        'categorical_features': cat_cols,
        'numerical_features': num_cols,
        'n_train': X_train_proc.shape[0],
        'n_features': X_train_proc.shape[1]
    }
    with open(output_dir / 'preprocess_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
