from flask import Flask, request, jsonify
import wbdata
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Global variables to store the trained model and scaler
model = None
feature_columns = ['GDP_growth', 'Inflation_CPI', 'E-commerce size', 'Cloud computing']

def initialize_model():
    """Initialize and train the model with the existing data"""
    global model
    
    try:
        # Set time range
        start_date = datetime.datetime(2019, 1, 1)
        end_date = datetime.datetime(2025, 12, 31)

        # Define World Bank indicators
        indicators = {
            'NY.GDP.MKTP.KD.ZG': 'GDP_growth',
            'FP.CPI.TOTL.ZG': 'Inflation_CPI',
            'FR.INR.RINR': 'Interest_rate',
        }

        # Fetch data for USA
        df = wbdata.get_dataframe(indicators, country='USA', data_date=(start_date, end_date))
        
        # Sort and reset index
        df = df.sort_index().reset_index()
        df.loc[5, 'GDP_growth'] = 2.8
        df.pop("Interest_rate")
        df = df.rename(columns={"date": "Date"})
        
        # Load additional data
        df_s = pd.read_csv('proj.csv')
            
        print(df_s.head())
        df['Date'] = df['Date'].astype(int)
        df_s['Date'] = df_s['Date'].astype(int)

        # Merge on 'Date'
        merged_df = df.merge(df_s, on='Date', how='left')

        # Prepare features
        X = merged_df[feature_columns]
        X = X.drop_duplicates(subset=feature_columns)
        
        # Prepare target (quarterly revenue)
        rev = []
        temp = []
        
        for i, v in enumerate(merged_df['Revenue']):
            if (i % 4 == 0 and i != 0):
                rev.append(temp)
                temp = []
            if i + 1 == len(merged_df["Revenue"]):
                rev.append(temp)
            temp.append(v)
        
        y = pd.DataFrame(rev, columns=["Q4", "Q3", "Q2", "Q1"])
        print(rev)
        
        # Train the model
        model = LinearRegression()
        model.fit(X, y)
        
        return True
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
   
    global model
    
    if model is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ['GDP_growth', 'Inflation_CPI', 'E-commerce_size', 'Cloud_computing']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
        
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'GDP_growth': float(data['GDP_growth']),
            'Inflation_CPI': float(data['Inflation_CPI']),
            'E-commerce size': float(data['E-commerce_size']),
            'Cloud computing': float(data['Cloud_computing'])
        }])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Format prediction results
        prediction_dict = {
            "Q1": round(float(prediction[0][3]/1000000000),2),  # Q1 is index 3
            "Q2": round(float(prediction[0][2]/1000000000),2),  # Q2 is index 2
            "Q3": round(float(prediction[0][1]/1000000000),2),  # Q3 is index 1
            "Q4": round(float(prediction[0][0]/1000000000),2)   # Q4 is index 0
        }
        print(prediction_dict)
        
        response = {
            "predictions": prediction_dict,
            "input_features": {
                "GDP_growth": data['GDP_growth'],
                "Inflation_CPI": data['Inflation_CPI'],
                "E-commerce_size": data['E-commerce_size'],
                "Cloud_computing": data['Cloud_computing']
            }
        }
        print(data['GDP_growth'])
        print(data['Cloud_computing'])
        print(data['E-commerce_size'])
        print(data['Inflation_CPI'])


        return jsonify(response)
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple sets of input features
    
    Expected JSON input:
    {
        "predictions": [
            {
                "GDP_growth": 3.0,
                "Inflation_CPI": 2.5,
                "E-commerce_size": 7000000000000,
                "Cloud_computing": 870000000000
            },
            {
                "GDP_growth": 2.8,
                "Inflation_CPI": 3.0,
                "E-commerce_size": 6500000000000,
                "Cloud_computing": 820000000000
            }
        ]
    }
    """
    global model
    
    if model is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({"error": "Invalid JSON format. Expected 'predictions' array"}), 400
        
        predictions_input = data['predictions']
        if not isinstance(predictions_input, list):
            return jsonify({"error": "'predictions' must be an array"}), 400
        
        results = []
        
        for i, pred_data in enumerate(predictions_input):
            # Validate required fields
            required_fields = ['GDP_growth', 'Inflation_CPI', 'E-commerce_size', 'Cloud_computing']
            missing_fields = [field for field in required_fields if field not in pred_data]
            
            if missing_fields:
                return jsonify({"error": f"Missing fields in prediction {i}: {missing_fields}"}), 400
            
            # Create input DataFrame
            input_data = pd.DataFrame([{
                'GDP_growth': float(pred_data['GDP_growth']),
                'Inflation_CPI': float(pred_data['Inflation_CPI']),
                'E-commerce size': float(pred_data['E-commerce_size']),
                'Cloud computing': float(pred_data['Cloud_computing'])
            }])
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Format prediction results
            prediction_dict = {
                "Q1": float(prediction[0][3]),
                "Q2": float(prediction[0][2]),
                "Q3": float(prediction[0][1]),
                "Q4": float(prediction[0][0])
            }
            
            results.append({
                "input_features": pred_data,
                "predictions": prediction_dict
            })
        
        return jsonify({"results": results})
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Initializing model...")
    port = int(os.environ.get('PORT', 5000))
    if initialize_model():
        print("Model initialized successfully!")
        print("\nAPI Endpoints:")
        print("- GET  /health - Health check")
        print("- POST /predict - Single prediction")
        print("- POST /predict_batch - Batch predictions")
        print("\nExample request to /predict:")
        print("""
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "GDP_growth": 3.0,
    "Inflation_CPI": 2.5,
    "E-commerce_size": 7000000000000,
    "Cloud_computing": 870000000000
  }'
        """)
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("Failed to initialize model. Please check your data files and dependencies.")