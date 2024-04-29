import gps
import time
import os
#from datetime import datetime
#from dateutil.parser import parse
#from dateutil import tz
i = 0

while True:
    # Listen on port 2947 (gpsd) of localhost
    session = gps.gps("localhost", "2947")
    session.stream(gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)
    while i == 0:
        
        
        report = session.next()
    		# Wait for a 'TPV' report and display the current time
    		# To see all report data, uncomment the line below
    		# print report
        if report['class'] == 'TPV':
            text_file = open("log.txt", "a")
            if hasattr(report, 'time'):
                #time = report.time
                #Eastern = tz.gettz('EDT')
                #time = d.astimezone(GMT)
                text_file.write("\nTime = %s \n" % report.time)
                #print "Time = %s" % report.time 
            if hasattr(report, 'lon'):
                text_file.write("Longitude = %s \n" % report.lon)
                #print "Longitude = %s" % report.lon
            if hasattr(report, 'lat'):
                text_file.write("Latitude = %s \n\n" % report.lat)
                #print "Latitude = %s" % report.lat
            
            text_file.close()
            i = i + 1
    i = 0
    #print "\n"
    time.sleep(10) 
    
