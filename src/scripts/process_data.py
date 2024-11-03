import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def read_file(filepath):
    return pd.read_csv(filepath, sep=",")


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

    # plt.tight_layout()
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


def process_data(filepath):
    data = read_file(filepath)
    #data_info, data_describe, data_class_frequency = explore_data(data)
    data_transformed = transform_data(data)
    return data_transformed

