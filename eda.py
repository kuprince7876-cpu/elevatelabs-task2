# Task 2: Exploratory Data Analysis (EDA) - Titanic Dataset
# Tools: Pandas, Matplotlib, Seaborn (Plotly optional)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 1. Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# 2. Summary statistics (mean, median, std, etc.)
num_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
print("\nSummary statistics:\n", df[num_cols].describe())
print("\nMedian:\n", df[num_cols].median())
print("\nSkewness:\n", df[num_cols].skew())
print("\nSurvival rate by Sex:\n", df.groupby("Sex")["Survived"].mean())
print("\nSurvival rate by Pclass:\n", df.groupby("Pclass")["Survived"].mean())
print("\nSurvival rate by Embarked:\n", df.groupby("Embarked")["Survived"].mean())

# 3. Histograms for numeric features
hist_cols = ["Age", "Fare", "SibSp", "Parch"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), hist_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Histogram of {col} (skew = {df[col].skew():.2f})")
plt.tight_layout()
plt.savefig("histograms.png", dpi=120)
plt.show()

# 4. Boxplots for numeric features (outliers)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
sns.boxplot(y=df["Age"], ax=axes[0], color="lightblue")
axes[0].set_title("Boxplot of Age")
sns.boxplot(y=df["Fare"], ax=axes[1], color="lightgreen")
axes[1].set_title("Boxplot of Fare")
sns.boxplot(data=df, x="Pclass", y="Fare", ax=axes[2])
axes[2].set_title("Fare by Pclass")
plt.tight_layout()
plt.savefig("boxplots.png", dpi=120)
plt.show()

# Outlier count using IQR rule
for col in ["Age", "Fare"]:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"Outliers in {col}: {n}")

# 5. Categorical features vs Survival
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="Sex", hue="Survived", ax=axes[0])
axes[0].set_title("Survival by Sex")
sns.countplot(data=df, x="Pclass", hue="Survived", ax=axes[1])
axes[1].set_title("Survival by Pclass")
sns.countplot(data=df, x="Embarked", hue="Survived", ax=axes[2])
axes[2].set_title("Survival by Embarked")
plt.tight_layout()
plt.savefig("survival_categories.png", dpi=120)
plt.show()

# 6. Correlation matrix
corr_df = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr()
print("\nCorrelation matrix:\n", corr_df.round(2))
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120)
plt.show()

# 7. Pairplot
pair_df = df[["Survived", "Pclass", "Age", "Fare"]].dropna()
g = sns.pairplot(pair_df, hue="Survived", palette="Set1", corner=True)
g.savefig("pairplot.png", dpi=100)
plt.show()

# 8. Optional interactive chart with Plotly
try:
    import plotly.express as px
    fig = px.scatter(df, x="Age", y="Fare", color=df["Survived"].astype(str),
                     title="Age vs Fare (coloured by Survived)")
    fig.write_html("age_vs_fare_plotly.html")
except ImportError:
    print("Plotly not installed - skipping interactive chart")

# 9. Patterns and inferences
print("""
INFERENCES
1. Women survived far more than men.
2. First class passengers survived more than third class passengers.
3. Age has missing values and Fare is highly right-skewed with many outliers.
4. Fare and Pclass are negatively correlated (higher class = higher fare).
5. Fare has a small positive correlation with Survived.
""")
