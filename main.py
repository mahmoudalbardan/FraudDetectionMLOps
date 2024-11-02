filepath = "/home/mahmoud/Documents/test ubisoft/data/creditcard.csv"

from src.scripts import process_data, data_explorer, data_transformer
from src.scripts import train_model, evaluate_model, save_model


def main():
    data = data_loader.read_file(filepath)
    #data_explorer
    data_transformed = data_transformer.transform_data(data)
    model = train_model.build_model(data_transformed)
    evaluate_model.evaluate_model(model, data_transformed)
    save_model.save_model(model)


if __name__ == "__main__":
    main()
    print("done")
