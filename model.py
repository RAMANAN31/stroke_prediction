import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# Load the improved dataset
df = pd.read_csv('improved_stroke_dataset_balanced.csv')

# Handle missing values (if any remaining, although this should already be cleaned)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# One-hot encode categorical variables (updated to match the new dataset)
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Check if all required columns are present
expected_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'] + \
    [col for col in df_encoded.columns if col.startswith(('gender_', 'ever_married_', 'work_type_', 'Residence_type_', 'smoking_status_'))]

missing_cols = set(expected_columns) - set(df_encoded.columns)
if missing_cols:
    print(f"Warning: The dataset is missing the following columns: {missing_cols}")
    raise ValueError("Dataset does not have the required columns.")

# Prepare feature set (X) and labels (y)
X = df_encoded.drop(columns=['stroke'])
y = df_encoded['stroke']

# Split into training and testing sets (no need for SMOTE as dataset is balanced)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier with default parameters
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
print('Confusion Matrix:')
print(conf_matrix)

# Save the trained model to a file
with open('stroke_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'stroke_model.pkl'")
