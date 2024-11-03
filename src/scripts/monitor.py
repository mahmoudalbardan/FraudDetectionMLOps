import joblib
from sklearn.metrics import recall_score

from process_data import read_file, transform_data
from utils import get_config, parse_args

def monitor(config):
    # load the existing model
    model = joblib.load(config["FILES"]["MODEL_PATH"])
    # load and transform new data
    new_data = read_file(config["FILES"]["DATA_PATH"])
    new_data_transformed = transform_data(new_data)
    # predict new data with existing model and check recall
    X_new = new_data_transformed.drop(columns=['Class'])
    y_true = new_data_transformed["Class"].values.tolist()
    y_pred = model.predict(X_new)
    y_pred = [1 if i == -1 else 0 for i in y_pred]
    recall = recall_score(y_true, y_pred)
    # send decision to retrain or not, truncate it each time
    with open(config["FILES"]["MONITORING_PATH"], 'w') as file:
        if recall < config["MONITOR"]["RECALL_THRESHOLD"]:
            file.write("retrain new model")
        else:
            file.write("keep existing model")


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.configuration)
    monitor(config)
