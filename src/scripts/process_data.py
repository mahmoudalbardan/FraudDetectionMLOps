import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

from google.cloud import storage
import pandas as pd
from io import StringIO


def read_file(gcs_bucket_name, gcs_filename):
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_filename)
    csv_content = blob.download_as_text()
    file = StringIO(csv_content)
    data = pd.read_csv(file, sep=",")
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


def process_data(gcs_bucket_name, gcs_filename):
    data = read_file(gcs_bucket_name, gcs_filename)
    # data_info, data_describe, data_class_frequency = explore_data(data)
    data_transformed = transform_data(data)
    return data_transformed
