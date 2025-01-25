from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load
from math import floor

app = Flask(__name__)

# Load the CSV file
GOAT_DATA_PATH = 'C:/GoatWeightEstimation/data/goats_features.csv'
goat_data = pd.read_csv(GOAT_DATA_PATH)

# Ensure 'Animal No.' is treated as a string and strip any extra spaces
goat_data['Animal No.'] = goat_data['Animal No.'].astype(str).str.strip()

# Preprocessor paths for each model
PREPROCESSOR_PATHS = {
    'XGBoost': 'C:/GoatWeightEstimation/model/preprocessor_xgb_regressor.pth',
    'Random Forest': 'C:/GoatWeightEstimation/model/preprocessor_random_forest.pth',
    'CatBoost': 'C:/GoatWeightEstimation/model/preprocessor_catboost.pth',
    'Gradient Boosting': 'C:/GoatWeightEstimation/model/preprocessor_gbr.pth',
    'AdaBoost': 'C:/GoatWeightEstimation/model/preprocessor_adaboost.pth',
    'Support Vector Regressor': 'C:/GoatWeightEstimation/model/preprocessor_svr.pth',
}

# Load preprocessors for each model
preprocessors = {key: load(path) for key, path in PREPROCESSOR_PATHS.items()}

# Model paths
MODEL_PATHS = {
    'XGBoost': 'C:/GoatWeightEstimation/model/xgboost.pth',
    'Random Forest': 'C:/GoatWeightEstimation/model/random_forest.pth',
    'CatBoost': 'C:/GoatWeightEstimation/model/catboost.pth',
    'Gradient Boosting': 'C:/GoatWeightEstimation/model/gbr.pth',
    'AdaBoost': 'C:/GoatWeightEstimation/model/adaboost.pth',
    'Support Vector Regressor': 'C:/GoatWeightEstimation/model/svr.pth',
}

# Load models for each regression technique
models = {key: load(path) for key, path in MODEL_PATHS.items()}

# Function to format age into years and months
def format_age(decimal_age):
    """Convert decimal age into a human-readable format (e.g., '2 years, 6 months')."""
    years = floor(decimal_age)  # Get the integer part (years)
    months = round((decimal_age - years) * 12)  # Convert the decimal part to months
    if years == 0:  # Less than a year
        return f"{months} months"
    elif months == 0:  # Exact number of years
        return f"{years} years old"
    else:  # Years and months
        return f"{years} years, {months} months"

@app.route('/')
def index():
    """Render the homepage with a dropdown of unique goat classes."""
    goat_classes = goat_data['Animal No.'].unique()
    return render_template('index.html', goat_classes=goat_classes)

@app.route('/predict', methods=['POST'])
def predict():
    """Process the user's selection and return predictions."""
    try:
        # Get the selected Animal No. from the form
        selected_animal_no = request.form['goat_description'].strip()

        # Filter the dataset to find the selected goat
        matching_rows = goat_data[goat_data['Animal No.'] == selected_animal_no]
        if matching_rows.empty:
            return f"Error: No goat found with Animal No. '{selected_animal_no}'.", 400

        selected_goat = matching_rows.iloc[0]

        # Prepare the features in DataFrame format
        input_features = pd.DataFrame([{
            'Heart Girth_cm': selected_goat['Heart Girth'],
            'Body Length_cm': selected_goat['Body Length'],
            'Age in Years': selected_goat['Age in Years'],
            'Sex': selected_goat['Sex'].strip().lower()
        }])

        # Initialize an empty dictionary for predictions
        predictions = {}

        # Process input features and predict using each model
        for name, model in models.items():
            # Use the corresponding preprocessor for this model
            preprocessor = preprocessors[name]
            processed_features = preprocessor.transform(input_features)

            # Make prediction
            predictions[name] = model.predict(processed_features)[0]

        # Render the result page with predictions
        return render_template(
            'result.html',
            predictions=predictions,
            goat_details={
                'class': selected_goat['Animal No.'],
                'body_height': selected_goat['Heart Girth'],
                'length': selected_goat['Body Length'],
                'age': selected_goat['Age in Years'],
                'formatted_age': format_age(selected_goat['Age in Years']),  # Use formatted age
                'sex': selected_goat['Sex'],
                'live_body_weight': selected_goat['Live Body Wt']
            }
        )
    except Exception as e:
        # Handle unexpected errors gracefully
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
