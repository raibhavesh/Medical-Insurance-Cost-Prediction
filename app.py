from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Path to the model file
Pkl_Filename = os.path.join(os.path.dirname(__file__), "rf_tuned.pkl")

# Try loading the model with joblib
try:
    model = joblib.load(Pkl_Filename)
except FileNotFoundError:
    print(f"Error: Model file {Pkl_Filename} not found.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if not model:
        return render_template('op.html', pred='Error: Model not loaded.')

    try:
        # Get the features from form input
        features = [int(x) for x in request.form.values()]
        final = np.array(features).reshape((1, 6))

        # Predict using the loaded model
        pred = model.predict(final)[0]

        # Check and render the prediction result
        if pred < 0:
            return render_template('op.html', pred='Error calculating Amount!')
        else:
            return render_template('op.html', pred='Expected amount is {0:.3f}'.format(pred))

    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template('op.html', pred='Error processing the input.')

if __name__ == '__main__':
    app.run(debug=True)
