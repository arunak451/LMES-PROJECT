import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
# Load dataset from CSV
data = pd.read_csv('dataset.csv')

# Display first 5 rows
print(data.head())

# Basic info about the dataset
print(data.info())

# Statistical summary
print(data.describe())
# Check for missing values
print(data.isnull().sum())

# Visualize missing values
import seaborn as sns
sns.heatmap(data.isnull(), cbar=False)

# Option 1: Drop rows with missing values
data_dropped = data.dropna()


# Option 2: Impute missing values
# For numerical data
num_imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
data[['numerical_col']] = num_imputer.fit_transform(data[['numerical_col']])

# For categorical data
cat_imputer = SimpleImputer(strategy='most_frequent')
data[['categorical_col']] = cat_imputer.fit_transform(data[['categorical_col']])

# For ordinal categories (where order matters)
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])

# For nominal categories (no order)
data = pd.get_dummies(data, columns=['gender'], drop_first=True)

# Alternatively using sklearn
# onehot_encoder = OneHotEncoder(drop='first', sparse=False)
# encoded = onehot_encoder.fit_transform(data[['nominal_col']])
# encoded_df = pd.DataFrame(encoded, columns=onehot_encoder.get_feature_names_out(['nominal_col']))
# data = pd.concat([data.drop(['nominal_col'], axis=1), encoded_df], axis=1)

# Standardize numerical columns
scaler = StandardScaler()
data[['numerical_col']] = scaler.fit_transform(data[['numerical_col']])

scaler = MinMaxScaler()
data[['numerical_col1', 'numerical_col2']] = scaler.fit_transform(data[['numerical_col1', 'numerical_col2']])

# Using Z-score
from scipy import stats
z_scores = stats.zscore(data['numerical_col'])
abs_z_scores = np.abs(z_scores)
outliers = (abs_z_scores > 3)

# Using IQR
Q1 = data['numerical_col'].quantile(0.25)
Q3 = data['numerical_col'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (data['numerical_col'] < lower_bound) | (data['numerical_col'] > upper_bound)


# Option 1: Remove outliers
data = data[~outliers]

# Option 2: Cap outliers
data['numerical_col'] = np.where(data['numerical_col'] > upper_bound, upper_bound,
                               np.where(data['numerical_col'] < lower_bound, lower_bound, 
                                       data['numerical_col']))


# For categorical data
cat_imputer = SimpleImputer(strategy='most_frequent')
data[['categorical_col']] = cat_imputer.fit_transform(data[['categorical_col']])

# Save processed data to CSV
data.to_csv('processed_data.csv', index=False)

