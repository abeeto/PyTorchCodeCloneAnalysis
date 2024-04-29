import sqlite3

con = sqlite3.connect('ChatBot.db')
cur = con.cursor()
for row in cur.execute('SELECT * FROM ChatBot '):
        print(row)
con.close()
