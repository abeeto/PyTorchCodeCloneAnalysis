import requests
import time
from datetime import datetime



api = '2ffd40362f6c4bdd050c1ad48eaa7891cb1e4890'

def get_zip_code():
	""" User input Zip Code for weather data
	"""
	print('Please enter your zip code:')
	zip_code = input()
	while len(zip_code) < 5 or len(zip_code) > 5:
		print('Invalid Zip Code')
		print('Please enter your zip code:')
		zip_code = input()
	return str(zip_code)

def get_weather_json(zip_code):
	""" GETs the JSON from the worldweatheronline API
	"""
	weather_url = 'http://api.worldweatheronline.com/free/v1/weather.ashx?q=' + str(zip_code) + '&format=json&num_of_days=1&key=' + str(api)
	




	#try: ####################################################################### ADDED TRY EXCEPT CASE HERE!!!
	weather_data = requests.get(weather_url)
	#except ConnectionError as e:
		#weather_data = "No Response"





	weather_json = weather_data.json()
	return weather_json

def get_temperature(json_data):
	""" Retrieves Temperature string from JSON
	"""
	temp = json_data['data']['current_condition'][0]['temp_F']
	return temp

def get_precipitation(json_data):
	""" Retrieves forcast code string from JSON
	"""
	forcast = json_data['data']['current_condition'][0]['precipMM'] #weatherCode
	return forcast

def get_windspeed(json_data):
	""" Retrieves wind MPH string from JSON
	"""
	wind = json_data['data']['current_condition'][0]['windspeedMiles']
	return str(wind)

def timenow():
	""" Gets the current time and posts it on the Torch every minute
	"""
	time_data = datetime.now().time().isoformat()
	url = 'https://api.spark.io/v1/devices/' + access_id + '/message'
	# print(type(time_data)) # str
	# print(time_data) # 00:42:55.923423423

	hour = int(time_data[0:2])
	minute = time_data[3:5]
	light = 'AM'

	if hour > 12:
		hour = hour - 12
		light = 'PM'
	elif hour == 00:
		hour = 12
	
	hour = str(hour)
	time = hour + ':' + minute + ' ' + light
	# print(time)
	data["message"] = time
	requests.post(url, data=data)
	
function = 'params'
access_id = '53ff6f065075535140441187'
access_token = 'a5c190e99b6803f74533e2876feb0c687e44cada'

url = 'https://api.spark.io/v1/devices/' + access_id + '/' + function
data = {'access_token': access_token}
array_of_RGB_values = [[0,0,0]]

def check_weather():
	""" API Pulls, Weather Data, and Weather codes
	"""
	weather_json = get_weather_json(zip_code_input)
	temp = get_temperature(weather_json)
	precipitation = get_precipitation(weather_json)
	wind = get_windspeed(weather_json)
	
	print(temp, 'F')
	print(precipitation, "MM Precipitation") #Weather Code
	print(wind, "MPH wind")
	#Setting arguments for weather torch

	args = upside_down(float(precipitation))


	last_RGB_values = array_of_RGB_values.pop()
	red, green, blue = location_specific_temp_spread(int(temp), 45, 85) #imputing temp, min temp range, max temp range
	change_colors(last_RGB_values[0], last_RGB_values[1], last_RGB_values[2], red, green, blue , args) #function call to change colors
	sleep(1) #function call to do 10 iterations (=10 minutes currently)

def upside_down(precipitation):
	""" If JSON returns rainy/snowy forcast, this function returns True. 
	"""
	if precipitation > 0.1: #weather_code > 142:
		return 'upside_down=1'
	return 'upside_down=0'

def sleep(refresh_rate):
	""" Time between Weather API calls (time interval between Torch color updates)
	"""
	sleep_time = refresh_rate
	while sleep_time > 0:
		print(str(sleep_time), 'minutes remaining')
		time.sleep(60) #60 seconds
		timenow()
		sleep_time-=1

def location_specific_temp_spread(temperature, mini, maxi):
	""" Since all cities have different temperature spreads, this function normalizes the lamp color
	temperature spreads
	"""
	spread = abs(maxi - mini)
	increment = spread // 11
	list_of_temp_ranges = []
	for i in range(11):
		list_of_temp_ranges.append(mini + (increment * i))

	return get_RGB(temperature, list_of_temp_ranges)

def get_RGB(temperature, lst):
	""" Gets the Torch Light colors based on the current Temperature
		temperature: the current temperature
		lst: the list of the temperature ranges for color change spread
		(ORIGINALLY: went from <50 to >= 95 with 11 ranges)
	"""
	red = 0
	green = 0
	blue = 0
	if temperature < lst[0]:
		blue = 255
	elif temperature < lst[1]:
		green = 128
		blue = 255
	elif temperature < lst[2]:
		green = 255
		blue = 255
	elif temperature < lst[3]:
		green = 255
		blue = 128
	elif temperature < lst[4]:
		green = 255
		blue = 0
	elif temperature < lst[5]:
		red = 128
		green = 255
	elif temperature < lst[6]:
		red = 255
		green = 255
	elif temperature < lst[7]:
		red = 255
		green = 128
	elif temperature < lst[8]:
		red = 255
		green = 102
		blue = 102
	elif temperature < lst[9]:
		red = 255
	elif temperature >= lst[10]:
		red = 153
	return red, green, blue

def change_colors(old_red, old_green, old_blue, new_red, new_green, new_blue, args):
	""" Torch color smooth transition between temperature settings
	"""
	print('Current Color Values:', old_red, old_green, old_blue)	
	print('Intended Color Values:', new_red, new_green, new_blue)	

	def color_value_changer(old_color, new_color):
		""" Increments old RGB value until it reaches the desired value
		"""
		if old_color < new_color:
			if old_color + 5 > new_color:
				old_color = new_color
			else:
				old_color += 5
		else:
			if old_color - 5 < new_color:
				old_color = new_color
			else:
				old_color -= 5
		return old_color
	
	while (old_red != new_red or old_blue != new_blue or old_green != new_green): #keep going if old_color does not match new_color
		
		if old_red != new_red:
			old_red = color_value_changer(old_red, new_red)
			
		if old_green != new_green:
			old_green = color_value_changer(old_green, new_green)

		if old_blue != new_blue:
			old_blue = color_value_changer(old_blue, new_blue)

		data['args'] = args + ',red_energy=' + str(old_red) + ',green_energy=' + str(old_green)+ ',blue_energy=' + str(old_blue)
		print(data['args'])
		requests.post(url, data=data)
		if old_red == new_red and old_blue == new_blue and old_green == new_green:
			break;

	array_of_RGB_values.append([new_red, new_green, new_blue])

zip_code_input = get_zip_code()
while True:
	try:
		check_weather()
	except requests.exceptions.ConnectionError as e:
		print("CONNECTION WAS LOST!", e)
		sleep(2)


