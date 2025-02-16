# import streamlit as st
# import numpy as np
# import joblib
# import tensorflow as tf

# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# tf.compat.v1.reset_default_graph()

# from tensorflow.keras.models import load_model

# tf.keras.utils.get_custom_objects().clear()

# model = load_model("weather_forecaster.h5", compile=False)

# model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# scaler_X = joblib.load("scaler_X.pkl") 
# scaler_y = joblib.load("scaler_y.pkl")  

# def main():
#     # Streamlit App Interface
#     st.title("🌦️ LSTM Weather Forecaster")

#     st.markdown("Enter weather conditions to predict if it will rain.")

#     # User input fields
#     temperature = st.number_input("Temperature (°C)", min_value=-30.0, max_value=50.0, value=25.0)
#     humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
#     wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)
#     pressure = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1013.0)
#     cloud_cover = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0)

#     # Make a prediction when the button is clicked
#     if st.button("Predict"):
#         # Convert user input into an array
#         input_data = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover]])

#         # Apply the feature scaler
#         input_scaled = scaler_X.transform(input_data)  # Scale input like training data

#         # Reshape input for LSTM (samples, time_steps, features)
#         input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))

#         # Make prediction
#         prediction_scaled = model.predict(input_reshaped)

#         # Inverse transform the prediction to get actual scale
#         prediction_actual = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

#         # Convert to binary classification
#         result = "🌧️ Rain Expected" if prediction_actual[0][0] > 0.5 else "☀️ No Rain Expected"

#         # Display result
#         st.subheader(f"Prediction: {result}")
#         st.write(f"Rain Probability: {prediction_actual[0][0]:.2%}")

# if __name__ == '__main__':
#     main()
















# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import os

# # Fix TensorFlow environment issues
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # Load trained model & scalers
# model = load_model("weather_forecaster.h5", compile=False)
# model.compile(optimizer="adam", loss="mse", metrics=["mae"])
# scaler_X = joblib.load("scaler_X.pkl")
# scaler_y = joblib.load("scaler_y.pkl")

# # Flask App
# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get input data from form
#         temperature = float(request.form["temperature"])
#         humidity = float(request.form["humidity"])
#         wind_speed = float(request.form["wind_speed"])
#         pressure = float(request.form["pressure"])
#         cloud_cover = float(request.form["cloud_cover"])

#         # Prepare input for model
#         input_data = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover]])
#         input_scaled = scaler_X.transform(input_data)
#         input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))

#         # Make prediction
#         prediction_scaled = model.predict(input_reshaped)
#         prediction_actual = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

#         # Convert to binary classification
#         result = "🌧️ Rain Expected" if prediction_actual[0][0] > 0.5 else "☀️ No Rain Expected"
#         rain_probability = f"{prediction_actual[0][0]:.2%}"

#         return render_template("index.html", prediction=result, probability=rain_probability)

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)






from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Fix TensorFlow environment issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load trained model & scalers
model = load_model("weather_forecaster.h5", compile=False)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "probability": None})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    temperature: float = Form(...),
    humidity: float = Form(...),
    wind_speed: float = Form(...),
    pressure: float = Form(...),
    cloud_cover: float = Form(...)
):
    try:
        # Prepare input for model
        input_data = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover]])
        input_scaled = scaler_X.transform(input_data)
        input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))

        # Make prediction
        prediction_scaled = model.predict(input_reshaped)
        prediction_actual = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

        # Convert to binary classification
        result = "🌧️ Rain Expected" if prediction_actual[0][0] > 0.5 else "☀️ No Rain Expected"
        rain_probability = f"{prediction_actual[0][0]:.2%}"

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": result,
                "probability": rain_probability,
            }
        )

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})











