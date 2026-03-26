def select_features(df):
    x = df.drop("Emp(2025)", axis=1)
    y = df["Emp(2025)"]
    return x, y