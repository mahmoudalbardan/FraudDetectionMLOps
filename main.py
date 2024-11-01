import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

filepath = "/home/mahmoud/Documents/test ubisoft/data/creditcard.csv"


def read_file(filepath):
    data = pd.read_csv(filepath, sep=",")
    return data


def explore_data(data):
    data_info = data.info()
    data_describe = data.describe()
    data_class_frequency = data['Class'].value_counts()
    features = data.columns[:-1]

    # histogram plots (univariate analysis)
    figure, axes = plt.subplots(4, 9, figsize=(15, 60))
    axes = axes.flatten()
    for j, feature in enumerate(features):
        sns.histplot(data[feature].values, bins=30, color="c", kde=True, ax=axes[j])
        axes[j].set_title("histogram of {feature}".format(feature=feature))
        axes[j].set_xlabel(feature)

    #plt.tight_layout()
    plt.show()
    return data_info, data_describe, data_class_frequency


def transform_data(data):
    data_transformed = data.copy()
    # applying log transform to reduce skewness of some features
    features = data.columns[:-1]
    for feature in features:
        if np.abs(data[feature].skew()) > 2:
            # applying log(1+x) because of errors of applying log on 0,
            # using sign of the element to deal with the negative values
            data_transformed[feature] = np.sign(data[feature].values) * \
                                        np.log1p(np.abs(data[feature].values))
    return data_transformed


def build_model(data_transformed):
    # contamination_level = len(data_transformed[data_transformed["Class"] == 1]) / len(data_transformed)
    X = data_transformed.drop(columns=['Class'])  # features
    X_scaled = StandardScaler().fit_transform(X)
    model = IsolationForest(n_estimators=100,
                            contamination=0.05,
                            random_state=42)
    model.fit(X_scaled)
    return model


def evaluate_model(model, data_transformed):
    X = data_transformed.drop(columns=['Class'])
    y_true = data_transformed["Class"].values.tolist()
    y_pred = model.predict(X)
    y_pred = [1 if i == -1 else 0 for i in y_pred]
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred)
    print("the recall rate is {rec}".format(rec=recall))
    print("the precision rate is {precision}".format(precision=precision))
    print("the f1_score rate is {f1}".format(f1=f1s))
    return recall, precision, f1s


def save_model(model):
    joblib.dump(model, 'models/fraud_detection_model.pkl')


def main():
    data = read_file(filepath)
    data_info, data_describe, data_class_frequency = explore_data(data)
    data_transformed = transform_data(data)
    model = build_model(data_transformed)
    recall, precision, f1s = evaluate_model(model, data_transformed)
    save_model(model)

if __name__ == "__main__":
    main()
    print("done")
