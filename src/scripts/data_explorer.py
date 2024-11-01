import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

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

    #plt.tight_layout()
    plt.show()
    return data_info, data_describe, data_class_frequency