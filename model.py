from sklearn.ensemble import RandomForestClassifier

def train_model(df):

    X = df[["ma20","ma50","rsi"]]
    y = df["target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=10,
        class_weight="balanced",  # 🔥 QUAN TRỌNG
        random_state=42
    )

    model.fit(X, y)
    return model
