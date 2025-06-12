"""from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('linear_model.pkl')  # Load the model using joblib

@app.route('/login')
def login():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Simple validation
    if username == "admin" and password == "12345":
        return f"Welcome, {username}!"
    else:
        return "Invalid credentials. Please try again."
@app.route('/')
def index():
    return render_template('mlindex.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form values
    val1 = request.form['bedrooms']
    val2 = request.form['bathrooms']
    val3 = request.form['floors']
    val4 = request.form['yr_built']

    # Convert values to NumPy array and float type
    arr = np.array([val1, val2, val3, val4], dtype=np.float64)
    
    # Make prediction
    pred = model.predict([arr])[0]  # Get the first (and only) prediction result

    # Return result to the UI
    return render_template('mlindex.html', prediction=f"${int(pred):,}")  # Format result as currency


if __name__ == '__main__':
    app.run(debug=True)
"""
"""

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('linear_model.pkl')  # Load pre-trained model

# Route for login page
@app.route('/login')
def login():
    return render_template('form.html')

# Login form handler
@app.route('/submit', methods=['POST'])
def submit():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Dummy user check (replace with real auth for production)
    if username == "admin" and password == "12345":
        return f"<h3>Welcome, {username}!</h3><a href='{url_for('index')}'>Go to Predictor</a>"
    else:
        return "<h4>Invalid credentials.</h4><a href='/login'>Try again</a>"

# Home route
@app.route('/')
def index():
    return render_template('mlindex.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        floors = float(request.form['floors'])
        yr_built = float(request.form['yr_built'])

        # Prepare input for model
        input_features = np.array([[bedrooms, bathrooms, floors, yr_built]])

        # Predict and format result
        predicted_price = model.predict(input_features)[0]
        formatted_price = f"${int(predicted_price):,}"

        return render_template('mlindex.html', prediction=formatted_price)
    except Exception as e:
        return f"<h4>Error occurred: {e}</h4><a href='/'>Go back</a>"

if __name__ == '__main__':
    app.run(debug=True)
"""
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the model
model = joblib.load('linear_model.pkl')

# 🔐 Login Page Route
@app.route('/login')
def login():
    return render_template('form.html')  # Your login form template

# 🔓 Handle Login Submission
@app.route('/submit', methods=['POST'])
def submit():
    username = request.form.get('username')
    password = request.form.get('password')

    if username == "admin" and password == "12345":
        session['logged_in'] = True
        return redirect(url_for('index'))  # Redirect to prediction page
    else:
        return "<h4>Invalid credentials.</h4><a href='/login'>Try again</a>"

# 🏠 Main Page (Prediction Form)
@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('mlindex.html')

# 📊 Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        floors = float(request.form['floors'])
        yr_built = float(request.form['yr_built'])

        input_features = np.array([[bedrooms, bathrooms, floors, yr_built]])
        predicted_price = model.predict(input_features)[0]
        formatted_price = f"${int(predicted_price):,}"

        return render_template('mlindex.html', prediction=formatted_price)
    except Exception as e:
        return f"<h4>Error occurred: {e}</h4><a href='/'>Go back</a>"

# 🔒 Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# 🚀 Run
if __name__ == '__main__':
    app.run(debug=True)
