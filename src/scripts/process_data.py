import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from google.cloud import storage
import pandas as pd
from io import StringIO

# Set Seaborn theme for plots
sns.set_theme(style="whitegrid")


def read_file(gcs_bucket_name, gcs_filename):
    """
    Read a CSV file from Google Cloud Storage.

    This function connects to Google Cloud Storage, retrieves the specified CSV file,
    and loads it into a Pandas DataFrame.

    Parameters
    ----------
    gcs_bucket_name : str
        The name of the Google Cloud Storage bucket.
    gcs_filename : str
        The name of the file within the bucket.

    Returns
    -------
    pd.DataFrame
        The contents of the CSV file as a Pandas DataFrame.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_filename)
    csv_content = blob.download_as_text()
    file = StringIO(csv_content)
    data = pd.read_csv(file, sep=",")
    return data


def explore_data(data):
    """
    Perform exploratory data analysis on a DataFrame.

    This function provides information about the DataFrame, descriptive statistics,
    and class frequency distribution. It also generates histogram plots for each feature.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to be explored.

    Returns
    -------
    tuple
        A tuple containing:
        - None: Info summary of the DataFrame (printed, not returned).
        - pd.DataFrame: Descriptive statistics of the DataFrame.
        - pd.Series: Frequency count of each class in the 'Class' column.
    """
    data_info = data.info()  # Print info summary
    data_describe = data.describe()
    data_class_frequency = data['Class'].value_counts()
    features = data.columns[:-1]

    # Histogram plots (univariate analysis)
    figure, axes = plt.subplots(4, 9, figsize=(15, 60))
    axes = axes.flatten()
    for j, feature in enumerate(features):
        sns.histplot(data[feature].values, bins=30, color="c", kde=True, ax=axes[j])
        axes[j].set_title(f"Histogram of {feature}")  # Use f-string for clarity
        axes[j].set_xlabel(feature)

    plt.show()
    return data_info, data_describe, data_class_frequency


def transform_data(data):
    """
    Transform features in the DataFrame to reduce skewness.

    This function applies a log transformation to features in the DataFrame
    that have a skewness greater than 2, which helps in normalizing the distribution.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to be transformed.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with transformed features.
    """
    data_transformed = data.copy()
    # Applying log transform to reduce skewness of some features
    features = data.columns[:-1]
    for feature in features:
        if np.abs(data[feature].skew()) > 2:
            # Apply log(1+x) to avoid issues with log(0)
            data_transformed[feature] = np.sign(data[feature].values) * \
                                        np.log1p(np.abs(data[feature].values))
    return data_transformed


def process_data(gcs_bucket_name, gcs_filename):
    """
    Process data by reading from Google Cloud Storage and transforming it.

    This function combines reading the data from Google Cloud Storage and
    transforming the features to reduce skewness.

    Parameters
    ----------
    gcs_bucket_name : str
        The name of the Google Cloud Storage bucket.
    gcs_filename : str
        The name of the file within the bucket.

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame ready for modeling.
    """
    data = read_file(gcs_bucket_name, gcs_filename)
    # data_info, data_describe, data_class_frequency = explore_data(data)  # Uncomment if exploration is needed
    data_transformed = transform_data(data)
    return data_transformed
