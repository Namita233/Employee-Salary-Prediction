import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Dummy dataset with correct column names
data = {
    'Age': [22, 25, 28, 35, 40],
    'Education_Level': [1, 2, 2, 3, 3],
    'Occupation': [1, 2, 3, 1, 2],
    'Experience': [0, 2, 4, 6, 8],
    'Hours': [6, 7, 8, 9, 10],
    'Salary': [15000, 20000, 25000, 30000, 35000]
}

df = pd.DataFrame(data)
X = df[['Age', 'Education_Level', 'Occupation', 'Experience', 'Hours']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'salary_model.pkl')
print("✅ Model trained and saved.")