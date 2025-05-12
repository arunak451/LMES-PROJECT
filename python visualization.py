import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('data (1).csv')

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

## 1. Pairplot to show relationships between all variables
print("Creating pairplot...")
sns.pairplot(df, diag_kind='kde')
plt.suptitle("Pairplot of All Variables", y=1.02)
plt.show()

## 2. Correlation Heatmap
print("Creating correlation heatmap...")
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()

## 3. Distribution of Grades with Socioeconomic Background
print("Creating grade distribution by socioeconomic score...")
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Socioeconomic Score', y='Grades', data=df, 
                hue='Grades', palette='viridis', size='Grades')
plt.title("Grades Distribution by Socioeconomic Background")
plt.show()

## 4. Study Hours vs. Sleep Hours colored by Grades
print("Creating study vs sleep hours with grades...")
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(x='Study Hours', y='Sleep Hours', data=df, 
                          hue='Grades', size='Grades', palette='coolwarm')
plt.title("Study Hours vs. Sleep Hours (Colored by Grades)")
plt.show()

## 5. Boxplots of Grades by Attendance Quartiles
print("Creating boxplots by attendance quartiles...")
df['Attendance Quartile'] = pd.qcut(df['Attendance (%)'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
plt.figure(figsize=(12, 6))
sns.boxplot(x='Attendance Quartile', y='Grades', data=df)
plt.title("Grade Distribution by Attendance Quartiles")
plt.show()

## 6. 3D Scatter Plot (requires mpl_toolkits)
print("Creating 3D scatter plot...")
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

xs = df['Study Hours']
ys = df['Sleep Hours']
zs = df['Grades']

ax.scatter(xs, ys, zs, c=df['Socioeconomic Score'], cmap='viridis', s=df['Attendance (%)']/2)
ax.set_xlabel('Study Hours')
ax.set_ylabel('Sleep Hours')
ax.set_zlabel('Grades')
plt.title("3D View: Study Hours, Sleep Hours, and Grades")
plt.show()

## 7. Hexbin Plot of Study Hours vs Grades
print("Creating hexbin plot...")
plt.figure(figsize=(12, 8))
plt.hexbin(df['Study Hours'], df['Grades'], gridsize=25, cmap='Blues')
plt.colorbar(label='Count in bin')
plt.xlabel('Study Hours')
plt.ylabel('Grades')
plt.title("Density of Study Hours vs. Grades")
plt.show()

## 8. Violin Plots for Each Variable
print("Creating violin plots...")
plt.figure(figsize=(12, 8))
df_melted = df.melt(value_vars=['Socioeconomic Score', 'Study Hours', 'Sleep Hours', 'Attendance (%)', 'Grades'])
sns.violinplot(x='variable', y='value', data=df_melted)
plt.title("Violin Plots of All Variables")
plt.xticks(rotation=45)
plt.show()

## 9. Parallel Coordinates Plot
print("Creating parallel coordinates plot...")
from pandas.plotting import parallel_coordinates

plt.figure(figsize=(12, 8))
parallel_coordinates(df.sample(100), 'Grades', colormap='viridis')  # Sampling for clarity
plt.title("Parallel Coordinates Plot (Sample of 100)")
plt.xticks(rotation=45)
plt.legend().remove()
plt.show()

## 10. Facet Grid of Relationships by Grade Bins
print("Creating facet grid...")
df['Grade Bins'] = pd.cut(df['Grades'], bins=5)
g = sns.FacetGrid(df, col='Grade Bins', col_wrap=3, height=4)
g.map(sns.scatterplot, 'Study Hours', 'Sleep Hours', alpha=0.7)
g.set_titles("Grade Range: {col_name}")
plt.suptitle("Study vs Sleep Hours Across Grade Ranges", y=1.05)
plt.show()