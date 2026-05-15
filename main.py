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

# Convert gender to numbers
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Fill missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)

# -------------------- GRAPH 1 --------------------
sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()
plt.clf()

# -------------------- GRAPH 2 --------------------
sns.countplot(x='Pclass', data=data)
plt.title("Passenger Class Count")
plt.savefig("pclass_count.png")
plt.show()
plt.clf()

# -------------------- GRAPH 3 --------------------
sns.histplot(data['Age'], bins=30)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.show()
plt.clf()

# -------------------- MACHINE LEARNING --------------------
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(accuracy)