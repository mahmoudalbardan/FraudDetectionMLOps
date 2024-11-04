import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from process_data import process_data
from utils import get_config, parse_args


def fit_model(data_transformed):
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


def save_model(model, model_path):
    joblib.dump(model, model_path)


def main(args):
    config = get_config(args.configuration)
    retrain = args.retrain
    if retrain == "false":
        data_transformed = process_data(config["FILES"]["GCS_BUCKET_NAME"],
                                        config["FILES"]["GCS_FIlE_NAME"])
    if retrain  == "true":
        data_transformed = process_data(config["FILES"]["GCS_BUCKET_NAME"],
                                        config["FILES"]["SAMPLE_GCS_FIlE_NAME"])
    model = fit_model(data_transformed)
    recall, precision, f1s = evaluate_model(model, data_transformed)
    save_model(model, config["FILES"]["MODEL_PATH"])


if __name__ == "__main__":
    args = parse_args()
    main(args)
