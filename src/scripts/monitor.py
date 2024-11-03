import os

import joblib
import pandas as pd
from sklearn.metrics import recall_score

from process_data import transform_data
from utils import get_config, parse_args


def load_updated_data():
    # load the data from GCP clous storage for example or S3 AWS
    # to run the code i'll put some data from
    updated_data = pd.DataFrame([[0, -1.3598071336738, -0.07278117330985, 2.53634673796914, 1.37815522427443,
                                  -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.098697901261051,
                                  0.363786969611213, 0.090794171978932, -0.551599533260813, -0.617800855762348,
                                  -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478,
                                  0.207971241929242, 0.025790580198559, 0.403992960255733, 0.251412098239705,
                                  -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.066928074914673,
                                  0.128539358273528, -0.189114843888824, 0.133558376740387, -0.021053053453822,
                                  149.62, 1],
                                 [0, -1.3598071336738, -0.07278117330985, 2.53634673796914, 1.37815522427443,
                                  -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.098697901261051,
                                  0.363786969611213, 0.090794171978932, -0.551599533260813, -0.617800855762348,
                                  -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478,
                                  0.207971241929242, 0.025790580198559, 0.403992960255733, 0.251412098239705,
                                  -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.066928074914673,
                                  0.128539358273528, -0.189114843888824, 0.133558376740387, -0.021053053453822,
                                  149.62, 1]],
                                columns=["Time"] + ["V" + str(i) for i in range(1, 29)] + ["Amount", "Class"])
    return updated_data


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
