from sklearn.metrics import recall_score, precision_score, f1_score


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
