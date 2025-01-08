import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# Load the dataset
data_path = 'Database/ChatGPT_ Quality of AI-Driven Services in Education  (Responses) - Form Responses 1 (1).csv'  # Update with your file path
df = pd.read_csv(data_path)

# Display the first few rows of the dataset
print("Initial Dataset:\n", df.head())


# Step 1: Convert Non-Numerical Columns to Numerical Values
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("\nDataset after Label Encoding:\n", df.head())


# Step 2: Data Cleaning
# Handling missing values
df.fillna(df.mean(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)



# Step 3: Descriptive Statistics
print("\nDescriptive Statistics:\n", df.describe())



# Step 4: Data Visualization
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Histogram for each column
df.hist(bins=20, figsize=(15, 10), edgecolor='black')
plt.tight_layout()
plt.show()

# Boxplot for each column
plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Boxplots of Columns')
plt.show()


# Step 5: Advanced Analysis
# Pairplot
sns.pairplot(df)
plt.show()



# Step 6: Logistic Regression Analysis
# Define target and features
target_column = 'Quality of Learning'  # Replace with actual target column if applicable
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Model evaluation
y_pred = log_reg.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importance = pd.Series(log_reg.coef_[0], index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", feature_importance)
