import requests
import json


def get_weather(city_name, api_key):
    base_url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    response = requests.get(base_url, params=params)
    weather_data = response.json()

    if response.status_code == 200:
        print(f"Weather in {city_name}: {weather_data['weather'][0]['description']}")
        print(f"Temperature: {weather_data['main']['temp']}°C")
        return weather_data['main']['temp']  # return the temperature value
    else:
        print(f"Failed to get weather data for {city_name}, status code: {response.status_code}")
        print(f"Response: {json.dumps(weather_data, indent=4)}")
        return None  # return None if request failed
