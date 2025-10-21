from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

model = joblib.load('model.pkl')
app = Flask(__name__,
template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_web', methods=['POST'])
def predict_web():
    data = {
        'Pclass': int(request.form['Pclass']),
        'Sex': request.form['Sex'],
        'Age': float(request.form['Age']),
        'SibSp': int(request.form['SibSp']),
        'Parch': int(request.form['Parch']),
        'Fare': float(request.form['Fare']),
        'Embarked': request.form['Embarked']
    }
    df = pd.DataFrame([data])
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    if 'Embarked' in df.columns:
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    for col in ['Embarked_Q', 'Embarked_S']:
        if col not in df.columns:
            df[col] = 0
    prediction = model.predict(df)[0]
    return f"<h2>Predicted Survival: {'Yes' if prediction==1 else 'No'}</h2><a href='/'>Try Again</a>"
import os
if __name__ == '__main__':
    port= int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0",port=port)