import pytest
from src.scripts.utils import get_config
from src.scripts.process_data import process_data
from src.scripts.train_model import fit_model, evaluate_model, save_model
import requests


def test_end_to_end(args):
    config = get_config(args.configuration)
    data_transformed = process_data(config["FILES"]["GCS_BUCKET_NAME"],
                                    config["FILES"]["SAMPLE_GCS_FILE_NAME"])

    model = fit_model(data_transformed)
    recall, precision, f1s = evaluate_model(model, data_transformed)
    save_model(model, config["FILES"]["MODEL_PATH"])
    sample_test = [0, -1.3598071336738, -0.07278117330985, 2.53634673796914, 1.37815522427443,
            -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.098697901261051,
            0.363786969611213, 0.090794171978932, -0.551599533260813, -0.617800855762348,
            -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478,
            0.207971241929242, 0.025790580198559, 0.403992960255733, 0.251412098239705,
            -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.066928074914673,
            0.128539358273528, -0.189114843888824, 0.133558376740387, -0.021053053453822,
            149.62]
    response = requests.post('http://localhost:5000/predict', json=sample_test)
    assert response.status_code == 200
    assert response.json() in [0, 1]


test_end_to_end()