def select_features(df):
    x = df.drop("Emp_2025", axis=1)
    y = df["Emp_2025"]
    return x, y