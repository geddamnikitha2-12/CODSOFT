import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Select important columns
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Convert male/female into numbers
data['Sex'] = data['Sex'].map({
    'male': 0,
    'female': 1
})

# Fill missing Age values
data['Age'].fillna(data['Age'].mean(), inplace=True)

# DATA VISUALIZATION

# Survival count graph
sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.show()

# Passenger class graph
sns.countplot(x='Pclass', data=data)
plt.title("Passenger Class Count")
plt.show()

# Age distribution graph
sns.histplot(data['Age'], bins=30)
plt.title("Age Distribution")
plt.show()

# MACHINE LEARNING

# Define input and output
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(accuracy)