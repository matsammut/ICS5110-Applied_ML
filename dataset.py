import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
print(plt.style.available)
# Load the dataset
file_path = 'imputed_dataset.csv'
data = pd.read_csv(file_path)

# Plot settings
plt.rcParams['figure.figsize'] = (20, 10)

# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Combine numerical and categorical columns
all_cols = numerical_cols.union(categorical_cols)
num_features = len(all_cols)

# Determine grid size
num_cols = 5  # Number of columns in the grid
num_rows = (num_features + num_cols - 1) // num_cols  # Calculate number of rows needed

# Plotting all features in a single grid
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 4 * num_rows))
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

for ax, col in zip(axes, all_cols):
    if col in numerical_cols:
        sns.histplot(data[col].dropna(), kde=True, bins=30, ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    elif col in ['gender', 'income']:
        # Create a pie chart for 'gender' and 'income'
        data[col].value_counts().plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Proportion of {col}')
        ax.set_ylabel('')  # Remove the y-label for pie charts
    else:
        # Use a bar chart for other categorical columns
        sns.countplot(y=data[col], order=data[col].value_counts().index, ax=ax)
        ax.set_title(f'Frequency of {col}')
        ax.set_xlabel('Count')
        ax.set_ylabel(col)

# Hide any unused subplots
for ax in axes[num_features:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

# Show the plot
plt.show()

# Create box plots for all columns
plt.figure(figsize=(20, 4 * num_rows))
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 4 * num_rows))
axes = axes.flatten()

for ax, col in zip(axes, all_cols):
    sns.boxplot(y=data[col], ax=ax)
    ax.set_title(f'Box Plot of {col}')
    ax.set_ylabel(col)

# Hide any unused subplots
for ax in axes[num_features:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

#######################################
from sklearn.preprocessing import LabelEncoder

# Encode categorical columns with label encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Calculate the correlation matrix
corr_matrix = data.corr()

# Create the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True)

# Add title and labels
plt.title('Correlation Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')

# Show the plot
plt.show()

# Display statistics for each column
print("\nColumn Statistics:")
print("-" * 50)
for col in all_cols:
    print(f"\n{col}:")
    print(f"Mean: {data[col].mean():.2f}")
    print(f"Median: {data[col].median():.2f}")
    print(f"Mode: {data[col].mode().iloc[0]}")

# Create a scatter plot for age vs income, colored by workclass
plt.figure(figsize=(12, 8))

# Get unique workclass values
unique_workclass = data['workclass'].unique()

# Create a color map for workclass categories
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_workclass)))

# Plot points for each workclass
for workclass, color in zip(unique_workclass, colors):
    mask = data['workclass'] == workclass
    plt.scatter(data.loc[mask, 'age'], 
               data.loc[mask, 'income'],
               color=color,
               label=f'Workclass {workclass}',
               alpha=0.6)

# Customize the plot
plt.xlabel('Age')
plt.ylabel('Income (0: ≤50K, 1: >50K)')
plt.title('Income vs Age by Workclass')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()



# Add small random noise to work


