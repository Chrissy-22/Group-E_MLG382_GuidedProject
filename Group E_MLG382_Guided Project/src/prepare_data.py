# python src/prepare_data.py
import pandas as pd
import numpy as np


def handle_missing_outliers(df):
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include=np.number):
        if col != "GPA":  # Exclude GPA from clipping
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower, upper)
    return df


if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df = handle_missing_outliers(train_df)
    test_df = handle_missing_outliers(test_df)

    test_df = test_df[train_df.columns] 

    train_df.to_csv("data/train_cleaned.csv", index=False)
    test_df.to_csv("data/test_cleaned.csv", index=False)
