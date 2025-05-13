#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv(r'C:\Users\k.a\OneDrive\Desktop\Documents\safety.csv')
df.head()

# Check for missing values
df.isnull().sum()

# Drop rows with missing values
df.dropna(inplace=True)

# Check for duplicate rows
df.duplicated().sum()

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Check for outliers
df.describe()

# Remove outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Check for data type errors
df.dtypes


# Data preprocessing
# Extract latitude and longitude from GPS coordinates
df['Latitude'] = df['GPS Coordinates'].str.extract(r'\((.*?),')[0].astype(float)
df['Longitude'] = df['GPS Coordinates'].str.extract(r'.*?,(.*?)\)')[0].astype(float)

# Convert date and time to datetime
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M')

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Device Type', 'Incident Type', 'Response Outcome', 'Day', 
                   'Location Type', 'Emergency Contact Status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create binary target for emergency (1 if Panic Button Activation is Yes or Incident Type is Assault)
df['Emergency'] = ((df['Panic Button Activation'] == 'Yes') | 
                   (df['Incident Type'] == label_encoders['Incident Type'].transform(['Assault'])[0])).astype(int)

# Feature engineering
df['Hour'] = df['DateTime'].dt.hour
df['Month'] = df['DateTime'].dt.month
df['Weekday'] = df['DateTime'].dt.weekday

##EDA PROCEESS

# Device Type Distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='Device Type', data=df, order=df['Device Type'].value_counts().index)
plt.title('Distribution of Device Types')
plt.xlabel('Count')
plt.ylabel('Device Type')
plt.show()

# Incident Type Distribution
plt.figure(figsize=(10, 6))
incident_counts = df['Incident Type'].value_counts()
plt.pie(incident_counts, labels=incident_counts.index, 
        startangle=90, colors=sns.color_palette('pastel'))
plt.title('Distribution of Incident Types')
plt.show()

# Response Time Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Response Time (minutes)'], bins=30, kde=True)
plt.title('Distribution of Response Times (minutes)')
plt.xlabel('Response Time (minutes)')
plt.ylabel('Count')
plt.show()

# Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution of Users')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Incidents by Hour of Day
plt.figure(figsize=(14, 6))
sns.countplot(x='Hour', data=df, hue='Emergency', palette='viridis')
plt.title('Incidents by Hour of Day (Emergency vs Non-Emergency)')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.legend(title='Emergency', labels=['No', 'Yes'])
plt.show()

# Incidents by Day of Week
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(14, 6))
sns.countplot(x='DayOfWeek', data=df, hue='Emergency', palette='coolwarm')
plt.title('Incidents by Day of Week (Emergency vs Non-Emergency)')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.xticks(ticks=range(7), labels=weekday_names)
plt.legend(title='Emergency', labels=['No', 'Yes'])
plt.show()

# Incidents by Month
plt.figure(figsize=(14, 6))
sns.countplot(x='Month', data=df, hue='Emergency', palette='magma')
plt.title('Incidents by Month (Emergency vs Non-Emergency)')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Emergency', labels=['No', 'Yes'])
plt.show()

# Heatmap of Incident Density
plt.figure(figsize=(12, 10))
sns.kdeplot(x=df['Longitude'], y=df['Latitude'], cmap='Reds', shade=True, bw_adjust=0.5)
plt.scatter(df['Longitude'], df['Latitude'], s=5, color='blue', alpha=0.5)
plt.title('Density Heatmap of Incidents')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# Response Time vs Incident Type
plt.figure(figsize=(14, 6))
sns.boxplot(x='Incident Type', y='Response Time (minutes)', data=df)
plt.title('Response Time Distribution by Incident Type')
plt.xticks(rotation=45)
plt.show()

# Age vs Incident Type
plt.figure(figsize=(14, 6))
sns.violinplot(x='Incident Type', y='Age', data=df, inner='quartile')
plt.title('Age Distribution by Incident Type')
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

##  Emergency Analysis

# Emergency vs Non-Emergency Comparison
emergency_df = df[df['Emergency'] == 1]
non_emergency_df = df[df['Emergency'] == 0]

# Device Type in Emergencies
plt.figure(figsize=(12, 6))
sns.countplot(y='Device Type', data=df, hue='Emergency', 
              order=df['Device Type'].value_counts().index)
plt.title('Device Type Usage in Emergencies vs Non-Emergencies')
plt.xlabel('Count')
plt.ylabel('Device Type')
plt.legend(title='Emergency', labels=['No', 'Yes'])
plt.show()

# Location Type in Emergencies
plt.figure(figsize=(12, 6))
sns.countplot(y='Location Type', data=df, hue='Emergency',
              order=df['Location Type'].value_counts().index)
plt.title('Location Type in Emergencies vs Non-Emergencies')
plt.xlabel('Count')
plt.ylabel('Location Type')
plt.legend(title='Emergency', labels=['No', 'Yes'])
plt.show()

# Emergency Contact Status
plt.figure(figsize=(12, 6))
sns.countplot(y='Emergency Contact Status', data=df, hue='Emergency',
              order=df['Emergency Contact Status'].value_counts().index)
plt.title('Emergency Contact Status in Emergencies vs Non-Emergencies')
plt.xlabel('Count')
plt.ylabel('Emergency Contact Status')
plt.legend(title='Emergency', labels=['No', 'Yes'])
plt.show()

# Select features for modeling
features = ['Device Type', 'Incident Type', 'Response Time (minutes)', 'Location Type', 
            'Age', 'Emergency Contact Status', 'Hour', 'Month', 'Weekday', 
            'Latitude', 'Longitude']
X = df[features]
y = df['Emergency']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Emergency Prediction with Multiple Algorithms

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Store results
    results[name] = {
        'accuracy': model.score(X_test_scaled, y_test),
        'report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'probabilities': y_prob
    }
    
    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred))

# Compare model accuracies
model_comparison = pd.DataFrame({k: [v['accuracy']] for k, v in results.items()}).T
model_comparison.columns = ['Accuracy']
model_comparison.sort_values('Accuracy', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=model_comparison.index, y='Accuracy', data=model_comparison)
plt.title('Model Comparison - Accuracy Scores')
plt.xticks(rotation=45)
plt.ylim(0.7, 1.0)
plt.show()

##Clustering for Incident Pattern Analysis

# Prepare data for clustering (focus on location and time)
cluster_features = ['Latitude', 'Longitude', 'Hour', 'Response Time (minutes)', 'Incident Type']
X_cluster = df[cluster_features]

# Scale data
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_cluster_scaled)
df['KMeans_Cluster'] = kmeans_labels


# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_cluster_scaled)
df['DBSCAN_Cluster'] = dbscan_labels

# Hierarchical Clustering
agg_cluster = AgglomerativeClustering(n_clusters=4)
agg_labels = agg_cluster.fit_predict(X_cluster_scaled)
df['Hierarchical_Cluster'] = agg_labels

# Evaluate clustering
print("K-Means Silhouette Score:", silhouette_score(X_cluster_scaled, kmeans_labels))
print("DBSCAN Silhouette Score:", silhouette_score(X_cluster_scaled, dbscan_labels))
print("Hierarchical Silhouette Score:", silhouette_score(X_cluster_scaled, agg_labels))

# Visualize clusters (using K-Means as example)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Longitude', y='Latitude', hue='KMeans_Cluster', 
                style='Incident Type', data=df, palette='viridis')
plt.title('Geographical Clusters of Incidents (K-Means)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

#Feature Importance Analysis
##Understanding which features contribute most to emergency predictions:
# Feature importance from Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Emergency Prediction')
plt.tight_layout()
plt.show()

#Analyzing temporal patterns in incidents:

# Resample by day to see incident trends
daily_incidents = df.set_index('DateTime').resample('D').size()
daily_emergencies = df[df['Emergency'] == 1].set_index('DateTime').resample('D').size()

plt.figure(figsize=(12, 6))
daily_incidents.plot(label='All Incidents')
daily_emergencies.plot(label='Emergencies')
plt.title('Daily Incident Trends')
plt.xlabel('Date')
plt.ylabel('Number of Incidents')
plt.legend()
plt.show()

# Hourly pattern of incidents
hourly_pattern = df.groupby('Hour').size()
hourly_emergencies = df[df['Emergency'] == 1].groupby('Hour').size()

plt.figure(figsize=(12, 6))
hourly_pattern.plot(label='All Incidents')
hourly_emergencies.plot(label='Emergencies')
plt.title('Hourly Incident Patterns')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Incidents')
plt.xticks(range(24))
plt.legend()
plt.show()


