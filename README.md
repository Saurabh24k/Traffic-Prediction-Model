# Traffic Prediction Model

## Overview
This project aims to predict the traffic volume on the I-94 Interstate highway. The prediction is based on several inputs, including weather conditions, date, and time. This model can aid in understanding traffic patterns and making informed decisions regarding traffic management and control.

## Dataset
The dataset used for training the model can be found at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume).

### Features:
- `holiday`: Indicator of a US National holiday or regional holiday.
- `temp`: Numeric Average temperature in Kelvin.
- `rain_1h`: Numeric Amount of rain in mm.
- `snow_1h`: Numeric Amount of snow in mm.
- `clouds_all`: Numeric Percentage of cloud cover.
- `weather_main`: Categorical Short textual description of the current weather.
- `weather_description`: Categorical Longer textual description of the current weather.
- `date_time`: DateTime Date and time of the hour of the data collected in CST/CDT.
- `traffic_volume`: Numeric Hourly I-94 ATR 301 reported westbound traffic volume.

### Data Preprocessing:
The script processes the date_time field to extract meaningful time-based features and applies one-hot encoding to categorical variables. Numerical features are standardized for optimal model performance.

## script.py - Model Training and Evaluation
This Python script performs the following functions:

### Data Preprocessing:
- Converts the `date_time` to datetime object and extracts the hour, day of the week, month, and year as separate features.
- Applies one-hot encoding to categorical features and scales numerical features using a `ColumnTransformer`.

### Model Training:
- Trains a RandomForestRegressor with pre-defined hyperparameters.
- Utilizes a pipeline to streamline the preprocessing and training process.

### Model Evaluation:
- Evaluates the model on a held-out test set using RMSE, MAE, and R-squared metrics.
- Outputs a histogram of residuals and a scatter plot of residuals vs. predicted values to visually assess the model fit.

### Running the Script:
To execute the training and evaluation process:
```bash
python script.py
```
## app.py - Flask Web Application
A Flask application is provided to serve model predictions through a RESTful API.

### Features:
- Loads the trained model and allows prediction via a POST request to `/predict`.
- Expects a JSON payload with model features and returns the predicted traffic volume.

### Usage:
To run the Flask application:
```bash
flask run
```
## make_prediction.py - Prediction Request
A utility script that sends a JSON-formatted POST request to the Flask web application to receive a traffic volume prediction.

### Running the Script:
To send a prediction request to the Flask app:

```bash
python make_prediction.py
```
## Evaluation Graphs
The model's performance is visualized through the following graphs:

### Histogram of Residuals
![Histogram of Residuals]([path/to/RvP.png](https://github.com/Saurabh24k/Traffic-Prediction-Model/blob/main/Histogram.png))
*The histogram illustrates the distribution of the residuals (predicted value minus the actual value), which should ideally resemble a normal distribution centered around zero.*

### Residuals vs. Predicted Values
![Residuals vs. Predicted Values]([path/to/Histogram.png](https://github.com/Saurabh24k/Traffic-Prediction-Model/blob/main/RvP.png))
*This scatter plot shows the residuals plotted against the predicted traffic volume. A random dispersion of points suggests that the model predictions are consistent across different values.*

## Requirements
This project requires Python 3 and the following Python libraries installed:

- NumPy
- Pandas
- Scikit-Learn
- Flask
- Matplotlib

You can install these packages using pip:

```bash
pip install numpy pandas scikit-learn flask matplotlib
