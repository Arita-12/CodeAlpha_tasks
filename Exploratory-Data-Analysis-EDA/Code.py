# ================================
# Exploratory Data Analysis (EDA)
# ================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
# Replace 'data.csv' with your dataset file
df = pd.read_csv("data.csv")

# ================================
# 1. Basic Information
# ================================

print("First 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# ================================
# 2. Check Missing Values
# ================================

print("\nMissing Values:")
print(df.isnull().sum())

# Visualize Missing Values
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# ================================
# 3. Check Duplicate Rows
# ================================

duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

# ================================
# 4. Data Types
# ================================

print("\nData Types:")
print(df.dtypes)

# Separate Numerical & Categorical Columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)

# ================================
# 5. Univariate Analysis
# ================================

# Histograms for Numerical Columns
df[num_cols].hist(figsize=(15,10), bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Boxplots for Outlier Detection
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Countplots for Categorical Columns
for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(x=df[col])
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# ================================
# 6. Correlation Analysis
# ================================

correlation_matrix = df[num_cols].corr()

plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ================================
# 7. Bivariate Analysis
# ================================

# Scatterplots Between Numerical Features
for col in num_cols:
    if col != num_cols[0]:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[num_cols[0]], y=df[col])
        plt.title(f"{num_cols[0]} vs {col}")
        plt.show()

# Pairplot
sns.pairplot(df[num_cols])
plt.show()

# ================================
# 8. Outlier Detection using IQR
# ================================

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\nOutliers in {col}: {len(outliers)}")

# ================================
# 9. Hypothesis Testing Example
# ================================

# Example: Correlation between two numerical columns

if len(num_cols) >= 2:
    corr_value = df[num_cols[0]].corr(df[num_cols[1]])
    
    print(f"\nCorrelation between {num_cols[0]} and {num_cols[1]}:")
    print(corr_value)

# ================================
# 10. Detect Class Imbalance
# ================================

# Replace 'target_column' with your target column name

target_column = "target_column"

if target_column in df.columns:
    print("\nTarget Variable Distribution:")
    print(df[target_column].value_counts())

    plt.figure(figsize=(6,4))
    sns.countplot(x=df[target_column])
    plt.title("Target Class Distribution")
    plt.show()

# ================================
# 11. Final Observations
# ================================

print("\nEDA Completed Successfully!")