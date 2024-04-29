import gps
import time
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
            if hasattr(report, 'time'):
                print "Time = %s" % report.time 
                #print report.time
            if hasattr(report, 'lon'):
                print "Longitude = %s" % report.lon
            if hasattr(report, 'lat'):
                print "Latitude = %s" % report.lat
            i = i + 1
    i = 0
    print "\n"
    time.sleep(5) 
    

            


	
