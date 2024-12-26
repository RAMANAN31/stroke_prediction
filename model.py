import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
df = pd.read_csv('improved_stroke_dataset_balanced.csv')

# Handle missing values (if any)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Define the expected columns for the model
expected_columns = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Male',
    'gender_Other', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
]

# Check if all expected columns are present
missing_cols = set(expected_columns) - set(df.columns)
if missing_cols:
    raise ValueError(f"The dataset is missing the following columns: {missing_cols}")

# Prepare feature set (X) and labels (y)
X = df[expected_columns]
y = df['stroke']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Save the trained model to a file
with open('stroke_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'stroke_model.pkl'")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
df = pd.read_csv('improved_stroke_dataset_balanced.csv')

# Handle missing values (if any)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Define the expected columns for the model
expected_columns = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Male',
    'gender_Other', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
]

# Prepare feature set (X) and labels (y)
X = df[expected_columns]
y = df['stroke']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle any class imbalance (even though it's a balanced dataset)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, min_samples_split=4, min_samples_leaf=2)
model.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Save the trained model to a file
with open('stroke_model_improved.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'stroke_model.pkl'")
"""
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
df = pd.read_csv('improved_stroke_dataset_balanced.csv')

# Handle missing values (if any)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Define the expected columns for the model
expected_columns = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Male',
    'gender_Other', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
]

# Prepare feature set (X) and labels (y)
X = df[expected_columns]
y = df['stroke']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle any class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train the SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

model.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
# Save the trained model to a file
with open('stroke_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'stroke_model.pkl'")
"""