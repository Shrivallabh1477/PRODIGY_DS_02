import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:\shree project\p\Task_2\test.csv')

# Print column names to verify
print(df.columns)

# Inspect the data
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # Drop 'Cabin' column as it has many missing values
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked values with the mode

# Convert categorical variables to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Exploratory Data Analysis (EDA)

# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', data=df)
plt.title('Count of Passengers by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='purchased', data=df)
plt.title('Purchase Rate by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Purchase Rate')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='purchased', data=df)
plt.title('Purchase Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Purchase Rate')
plt.show()

# Correlation Analysis
# Drop non-numeric columns (e.g., 'Name', 'Ticket', 'PassengerId', etc.)
df_cleaned = df.drop(columns=['Name', 'Ticket', 'PassengerId'], errors='ignore')

# Select numeric columns for correlation analysis
numeric_df = df_cleaned.select_dtypes(include=['float64', 'int64'])

# Compute and plot correlation matrix
plt.figure(figsize=(12, 8))
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
