from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

# Take a look at the data structure
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts()) # Shows which categories exist and frequency of each
print(housing.describe()) # Details about numeric aspects

# Plot histogram for each attribute
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize= (12,8))
plt.show()

# Create a test set
import numpy as np
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) # Shuffle indices
    test_set_size = int((len(data))*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = shuffle_and_split_data(housing,0.2)
print(len(train_set), len(test_set))

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)