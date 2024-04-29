from stem import Signal
from stem.control import Controller
import requests
from torrequest import TorRequest

tr=TorRequest(password='blahblahblah')
tr.reset_identity() #Reset Tor
response= tr.get('http://ipecho.net/plain')
print ("New Ip Address",response.text)
#------------------------------------------------------------
response= requests.get('http://ipecho.net/plain')
print ("My Original IP Address:",response.text)
#------------------------------------------------------------

with Controller.from_port(port = 9051) as controller:
	controller.authenticate(password='13711731@gmail.com') 
	print("Success!")
	controller.signal(Signal.NEWNYM)
	print("New Tor connection processed")
response= requests.get('http://ipecho.net/plain')
print ("IP Address after success s:",response.text)
