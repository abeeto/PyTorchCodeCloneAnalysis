import sqlite3

conn = sqlite3.connect("test.db")
cur = conn.cursor()

sql = "select * from tbl_CameraList"
cur.execute(sql)
rows = cur.fetchall()
for row in rows:
    print(row)

conn.close()



