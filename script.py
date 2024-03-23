import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
import matplotlib.pyplot as plt

# Loading the dataset
file_path = 'Metro_Interstate_Traffic_Volume.csv'
traffic_data = pd.read_csv(file_path)

# Data preprocessing
traffic_data['date_time'] = pd.to_datetime(traffic_data['date_time'])
traffic_data['hour'] = traffic_data['date_time'].dt.hour
traffic_data['day_of_week'] = traffic_data['date_time'].dt.dayofweek
traffic_data['month'] = traffic_data['date_time'].dt.month
traffic_data['year'] = traffic_data['date_time'].dt.year
traffic_data = traffic_data.drop('date_time', axis=1)

# Defining categorical and numerical features
categorical_cols = ['holiday', 'weather_main', 'weather_description']
numerical_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'day_of_week', 'month', 'year']

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ])

# Splitting the dataset
X = traffic_data.drop('traffic_volume', axis=1)
y = traffic_data['traffic_volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Creatign the model with the best parameters
best_rf_model = RandomForestRegressor(max_depth=10, min_samples_split=5, n_estimators=200, random_state=32)

# Creating the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', best_rf_model)])

# Fitting the pipeline on the training data
pipeline.fit(X_train, y_train)

# Serializing the model to a file
model_filename = 'traffic_volume_prediction_model.joblib'
dump(pipeline, model_filename)
print(f"Model saved to {model_filename}")

# Loading the model from the file
loaded_model = load(model_filename)
print("Model loaded successfully")

# Predicting and evaluating the model
predictions = loaded_model.predict(X_test)

# Calculating evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Printing evaluation metrics
print(f"RMSE with Loaded Model: {rmse:.2f}")
print(f"MAE with Loaded Model: {mae:.2f}")
print(f"R-squared with Loaded Model: {r2:.2f}")

# Model Evaluation Plots

# Plotting Residuals vs. Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(predictions, y_test - predictions, alpha=0.5)
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Traffic Volume')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True)
plt.show()

# Histogram of the residuals
plt.figure(figsize=(10, 6))
plt.hist(y_test - predictions, bins=30, edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
