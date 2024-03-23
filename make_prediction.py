import requests

url = 'http://127.0.0.1:5000/predict'

data = {
    'temp': 288.28,
    'rain_1h': 0.0,
    'snow_1h': 0.0,
    'clouds_all': 40,
    'holiday': 'None',
    'weather_main': 'Clear',
    'weather_description': 'scattered clouds',
    'hour': 9,
    'day_of_week': 1,
    'month': 10,
    'year': 2012
}

response = requests.post(url, json=data)
prediction = response.json()

if 'error' in prediction:
    print(f"An error occurred: {prediction['error']}")
else:
    traffic_volume = prediction['prediction'][0]
    print(f"The predicted traffic volume is approximately {traffic_volume:.2f} vehicles per hour.")