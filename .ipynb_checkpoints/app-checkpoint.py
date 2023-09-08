import os
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model upon starting the application
model_path = os.path.join(os.getcwd(), 'model.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive post request and predict the result."""
    
    # Extract input features from the form
    input_features = [int(x) for x in request.form.values()]
    array_features = np.array([input_features])
    
    # Make prediction
    prediction = model.predict(array_features)
    predicted_salary = round(prediction[0], 2)
    
    return render_template('index.html', text=f'Çalışanın maaşı {predicted_salary} lira olmalı')

if __name__ == '__main__':
    app.run(debug=True)
