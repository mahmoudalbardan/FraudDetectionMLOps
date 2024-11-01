import joblib


def save_model(model):
    joblib.dump(model, 'src/model/fraud_detection_model.pkl')
