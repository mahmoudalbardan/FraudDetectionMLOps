from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def build_model(data_transformed):
    # contamination_level = len(data_transformed[data_transformed["Class"] == 1]) / len(data_transformed)
    X = data_transformed.drop(columns=['Class'])  # features
    X_scaled = StandardScaler().fit_transform(X)
    model = IsolationForest(n_estimators=100,
                            contamination=0.05,
                            random_state=42)
    model.fit(X_scaled)
    return model


