import requests


api = '2ffd40362f6c4bdd050c1ad48eaa7891cb1e4890'


def get_weather_json(zip_code):
	weather_url = 'http://api.worldweatheronline.com/free/v1/weather.ashx?q=' + str(zip_code) + '&format=json&num_of_days=1&key=' + str(api)
	# print(weather_url)
	weather_data = requests.get(weather_url)
	weather_json = weather_data.json()
	return weather_json

def get_forcast(json_data):
	forcast = json_data['data']['current_condition'][0]['weatherCode']
	return forcast

def get_location_info(zip):
	url = 'http://maps.googleapis.com/maps/api/geocode/json?address=' + str(zip) + '&sensor=true'
	location_data = requests.get(url)
	location_json = location_data.json()
	return location_json['results'][0]['formatted_address']

counter = 99950
while counter > 0:
	weather_json = get_weather_json(counter)
	# print(counter)
	# print(weather_json)
	if 'error' in weather_json['data'].keys():
		counter-=1
		continue
	weather_code = get_forcast(weather_json)
	if int(weather_code) > 125:
		print(weather_code)
		print(counter)
		print(get_location_info(counter))
		break
	counter-=1
