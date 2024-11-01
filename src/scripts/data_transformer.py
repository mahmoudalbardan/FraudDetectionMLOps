import numpy as np

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