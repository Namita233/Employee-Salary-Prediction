from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('salary_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        education = int(request.form['education'])
        occupation = int(request.form['occupation'])
        experience = int(request.form['experience'])
        hours = int(request.form['hours'])

        # Prepare input exactly as model expects
        df = pd.DataFrame([[age, education, occupation, experience, hours]],
                          columns=['Age', 'Education_Level', 'Occupation', 'Experience', 'Hours'])

        prediction = model.predict(df)[0]

        return render_template('index.html', prediction=f'Predicted Salary: ₹{int(prediction)}')

    except Exception as e:
        return render_template('index.html', prediction=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)