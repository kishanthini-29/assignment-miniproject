from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

def preprocess_input(pclass, age, family_size, sex):
    pclass = int(pclass)
    age = float(age)
    family_size = int(family_size)
    sex_encoded = 1 if sex == 'male' else 0
    
    # Create the input DataFrame
    input_df = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'FamilySize': [family_size],
        'Fare': [0.0],       
        'Parch': [0],         
        'PassengerId': [0],  
        'SibSp': [0],        
        'Sex_female': [1 - sex_encoded], 
        'Sex_male': [sex_encoded]
    })
    
    return input_df

def predict_survival(pclass, age, family_size, sex):
    input_df = preprocess_input(pclass, age, family_size, sex)
    input_df = input_df[[
        'Pclass', 'Age', 'FamilySize', 'Fare', 'Parch', 'PassengerId', 'SibSp', 'Sex_female', 'Sex_male'
    ]]
    
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        return "Prediction: Survived"
    else:
        return "Prediction: Did not survive"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            pclass = request.form['pclass']
            age = request.form['age']
            family_size = request.form['family_size']
            sex = request.form['sex']
            
            result = predict_survival(pclass, age, family_size, sex)
            return result
        except Exception as e:
            print("An exception occurred:", e)
            return "An error occurred"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
