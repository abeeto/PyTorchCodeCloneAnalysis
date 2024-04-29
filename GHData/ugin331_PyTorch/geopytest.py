from geopy.geocoders import Nominatim


geolocator = Nominatim(user_agent="testAgent")
location = geolocator.reverse("52.509669, 13.376294")

