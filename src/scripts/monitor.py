import os

import joblib
import pandas as pd
from sklearn.metrics import recall_score

from process_data import transform_data, read_file
from utils import get_config, parse_args


def load_updated_data():
    # load the data from GCP clous storage for example or S3 AWS
    # to run the code i'll put some selected data samples of size 18Kb that will be uploaded on github
    sample_data = read_file(config["FILES"]["SAMPLE_DATA_PATH"])
    sample_data["Class"] = 1 # change target in order to drop down the recall and test retraining new model
    return sample_data


def monitor(config):
    # load the existing model
    model = joblib.load(config["FILES"]["MODEL_PATH"])
    # load and transform new data
    new_data = load_updated_data()
    new_data_transformed = transform_data(new_data)
    # predict new data with existing model and check recall
    X_new = new_data_transformed.drop(columns=['Class'])
    y_true = new_data_transformed["Class"].values.tolist()
    y_pred = model.predict(X_new)
    y_pred = [1 if i == -1 else 0 for i in y_pred]
    recall = recall_score(y_true, y_pred)
    # send decision to retrain or not, truncate it each time
    with open(config["FILES"]["MONITORING_PATH"], 'w') as file:
        if recall < float(config["MONITOR"]["RECALL_THRESHOLD"]):
            file.write("retrain new model")
        else:
            file.write("keep existing model")


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.configuration)
    monitor(config)
