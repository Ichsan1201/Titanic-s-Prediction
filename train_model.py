import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Memuat dataset Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Memilih fitur yang relevan
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]

# Menangani missing value
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encoding kategorikal
df['Sex'] = df['Sex'].map({
    'male': 1,
    'female': 0
})

df['Embarked'] = df['Embarked'].map({
    'C': 0,
    'Q': 1,
    'S': 2
})

# Feature dan target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Training model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Simpan model
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model berhasil disimpan sebagai titanic_model.pkl")