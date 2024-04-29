import urllib.request
import xml.etree.ElementTree as ET
import sqlite3

naiz_url = 'http://naiz.re.kr:8001/camera/list.cgi?id=admin&password=admin&key=all&method=get'
conn = sqlite3.connect("test.db")
cur = conn.cursor()

file = urllib.request.urlopen( naiz_url ).read().decode('euc-kr')
root = ET.fromstring(file)
iCount = 0

for child in root :
    for sub in child :
        for item in sub :
            if (item.tag == 'Key') :      
                UniqueKey = item.text
            if (item.tag == 'Name') :      
                Name = item.text
            if (item.tag == 'Address') :      
                IP_Addr = item.text
            if (item.tag == 'RTSP_URL1') :    
                RTSP_URL1 = item.text  
            if (item.tag == 'RTSP_URL2') :      
                RTSP_URL2 = item.text  
        print("UniqueKey = " + UniqueKey)
        print("Name = " + Name)
        print("Address = " + IP_Addr)
        print("RTSP_URL #1 = " + RTSP_URL1)
        print("RTSP_URL #2 = " + RTSP_URL2)
        iCount = iCount + 1
# sqlite test.db 에 해당내용 저장        
        try :
            sql_stmt = "insert into tbl_CameraList(seq,name,ip_addr,rtsp_url1,rtsp_url2) values(?,?,?,?,?)"
            cur.execute( sql_stmt,(int(UniqueKey),Name,IP_Addr,RTSP_URL1,RTSP_URL2))
            conn.commit()
        except :
            print("중복키 발생")

print("\n전체갯수 = " + str(iCount))
conn.close()

# name = root.find('Camera').find('CameraList').find('CameraListItem').find('Name')
#print( file )