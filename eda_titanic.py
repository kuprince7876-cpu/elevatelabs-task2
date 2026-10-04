import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create directory for saving plots if it doesn't exist
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# Load the dataset
print("Loading Titanic dataset...")
df = pd.read_csv('titanic_original.csv')
print(f"Dataset shape: {df.shape}")

# Display basic information
print("\n=== Dataset Info ===")
print(df.info())

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Summary Statistics ===")
print(df.describe(include='all'))

# 1. Generate summary statistics (mean, median, std, etc.)
print("\n=== Summary Statistics for Numeric Features ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
summary_stats = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
print(summary_stats)

# 2. Create histograms and boxplots for numeric features
print("\nCreating histograms and boxplots...")
n_numeric = len(numeric_cols)
n_cols = 3
n_rows = (n_numeric + n_cols - 1) // n_cols

# Histograms
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
fig.suptitle('Histograms of Numeric Features', fontsize=16)
axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

for i, col in enumerate(numeric_cols):
    if i < len(axes):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Histogram of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

# Hide unused subplots
for i in range(len(numeric_cols), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(f'{plots_dir}/histograms.png', dpi=300, bbox_inches='tight')
plt.close()

# Boxplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
fig.suptitle('Boxplots of Numeric Features', fontsize=16)
axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

for i, col in enumerate(numeric_cols):
    if i < len(axes):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel(col)

# Hide unused subplots
for i in range(len(numeric_cols), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(f'{plots_dir}/boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Use pairplot/correlation matrix for feature relationships
print("\nCreating correlation matrix and pairplot...")

# Correlation matrix for numeric features
corr_matrix = df[numeric_cols].corr()

# Plot correlation matrix heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig(f'{plots_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Pairplot for a subset of features (to avoid too many plots)
subset_cols = ['survived', 'pclass', 'age', 'fare', 'sibsp', 'parch']
subset_cols = [col for col in subset_cols if col in df.columns]
if len(subset_cols) > 1:
    pairplot_data = df[subset_cols].dropna()
    if len(pairplot_data) > 0:
        sns.pairplot(pairplot_data, hue='survived' if 'survived' in subset_cols else None, 
                     diag_kind='hist', plot_kws={'alpha': 0.6})
        plt.suptitle('Pairplot of Selected Features', y=1.02)
        plt.savefig(f'{plots_dir}/pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()

# 4. Identify patterns, trends, or anomalies in the data
print("\n=== Identifying Patterns and Anomalies ===")

# Missing values analysis
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})
print("\nMissing Values:")
print(missing_df[missing_df['Missing Values'] > 0])

# Survival rate by different categories
print("\n=== Survival Analysis ===")
if 'survived' in df.columns and 'sex' in df.columns:
    survival_by_sex = df.groupby('sex')['survived'].mean()
    print("\nSurvival rate by sex:")
    print(survival_by_sex)

if 'survived' in df.columns and 'pclass' in df.columns:
    survival_by_class = df.groupby('pclass')['survived'].mean()
    print("\nSurvival rate by passenger class:")
    print(survival_by_class)

if 'survived' in df.columns and 'embarked' in df.columns:
    survival_by_embarked = df.groupby('embarked')['survived'].mean()
    print("\nSurvival rate by embarkation point:")
    print(survival_by_embarked)

# 5. Make basic feature-level inferences from visuals
print("\n=== Feature-Level Inferences ===")

# Save inferences to a text file
with open('feature_inferences.txt', 'w') as f:
    f.write("FEATURE-LEVEL INFERENCES FROM EDA\n")
    f.write("=" * 40 + "\n\n")
    
    # Age inferences
    if 'age' in df.columns:
        age_mean = df['age'].mean()
        age_median = df['age'].median()
        f.write(f"Age: Mean = {age_mean:.2f}, Median = {age_median:.2f}\n")
        if age_mean > age_median:
            f.write("  -> Age distribution is right-skewed (more younger passengers)\n")
        else:
            f.write("  -> Age distribution is left-skewed or symmetric\n")
    
    # Fare inferences
    if 'fare' in df.columns:
        fare_mean = df['fare'].mean()
        fare_median = df['fare'].median()
        f.write(f"\nFare: Mean = {fare_mean:.2f}, Median = {fare_median:.2f}\n")
        if fare_mean > fare_median:
            f.write("  -> Fare distribution is right-skewed (few passengers paid very high fares)\n")
        else:
            f.write("  -> Fare distribution is left-skewed or symmetric\n")
    
    # Survival inferences
    if 'survived' in df.columns:
        survival_rate = df['survived'].mean() * 100
        f.write(f"\nOverall Survival Rate: {survival_rate:.2f}%\n")
        
        if 'sex' in df.columns:
            female_survival = df[df['sex'] == 'female']['survived'].mean() * 100
            male_survival = df[df['sex'] == 'male']['survived'].mean() * 100
            f.write(f"Female Survival Rate: {female_survival:.2f}%\n")
            f.write(f"Male Survival Rate: {male_survival:.2f}%\n")
            f.write("  -> Females had significantly higher survival rate than males\n")
        
        if 'pclass' in df.columns:
            for pclass in sorted(df['pclass'].unique()):
                class_survival = df[df['pclass'] == pclass]['survived'].mean() * 100
                f.write(f"Class {pclass} Survival Rate: {class_survival:.2f}%\n")
            f.write("  -> Survival rate decreases with passenger class (1st > 2nd > 3rd)\n")

print("Feature inferences saved to 'feature_inferences.txt'")

# 6. Create visualizations for categorical features
print("\nCreating visualizations for categorical features...")
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    n_cat = len(categorical_cols)
    n_cols_cat = 2
    n_rows_cat = (n_cat + n_cols_cat - 1) // n_cols_cat
    
    fig, axes = plt.subplots(n_rows_cat, n_cols_cat, figsize=(15, 5*n_rows_cat))
    fig.suptitle('Count Plots of Categorical Features', fontsize=16)
    axes = axes.flatten() if n_rows_cat > 1 else [axes] if n_rows_cat == 1 else axes
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            value_counts = df[col].value_counts()
            axes[i].bar(range(len(value_counts)), value_counts.values)
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[i].set_title(f'Count Plot of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
    
    # Hide unused subplots
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/categorical_counts.png', dpi=300, bbox_inches='tight')
    plt.close()

# Save processed dataset (for reference)
df.to_csv('titanic_eda_processed.csv', index=False)
print("\nProcessed dataset saved as 'titanic_eda_processed.csv'")

print("\nEDA Complete! Check the 'plots' directory for visualizations.")
print("Key files created:")
print("- plots/histograms.png")
print("- plots/boxplots.png")
print("- plots/correlation_matrix.png")
print("- plots/pairplot.png")
print("- plots/categorical_counts.png")
print("- feature_inferences.txt")
print("- titanic_eda_processed.csv")
