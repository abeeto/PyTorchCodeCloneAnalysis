import os
from typing import List
import sqlite3
import io

import numpy as np


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("ARRAY", convert_array)


class FaceRecognitionDataBase(object):

    def __init__(self, database_file) -> None:

        if not os.path.exists(database_file):
            self.conn = sqlite3.connect(database_file, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
            self.create_database()
        else:
            self.conn = sqlite3.connect(database_file, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        #self.database_fike = database_file


    #def connect(self):
    #    if not os.path.exists(self.database_file):
    #        self.conn = sqlite3.connect(self.database_file, detect_types=sqlite3.PARSE_DECLTYPES)
    #        self.create_database()
    #    else:
    #        self.conn = sqlite3.connect(self.database_file, detect_types=sqlite3.PARSE_DECLTYPES)

    
    def create_database(self):
        cursor = self.conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS INFO_FILE (PERSON_ID INTEGER PRIMARY KEY AUTOINCREMENT, INFO TEXT, FEATURE_VECTOR ARRAY NOT NULL);""")
        self.conn.commit()
        cursor.close()

    def add_persons(self, info, features):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO INFO_FILE (INFO, FEATURE_VECTOR) VALUES (?, ?)", (info, features, ))
        self.conn.commit()
        cursor.close()

    def get_person_by_id(self, person_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM INFO_FILE WHERE PERSON_ID = ?", (person_id, ))
        vectors = cursor.fetchall()
        cursor.close()
        return vectors

    def get_all_id(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM INFO_FILE")
        persons = cursor.fetchall()
        cursor.close()
        return persons

    def get_latest_id(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM INFO_FILE ORDER BY PERSON_ID DESC LIMIT 1;")
        persons = cursor.fetchall()[0]
        cursor.close()
        return persons

    def delete_by_id(self, person_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM INFO_FILE WHERE PERSON_ID = ?", (person_id, ))
        self.conn.commit()
        cursor.close()

    def close(self):
        self.conn.close()

    
#database = FaceRecognitionDataBase("info.db")

#for _ in range(100):
#    random_vector = np.random.rand(2, 100)
#    database.add_persons(random_vector)


#persons = database.get_all_id()
#print(persons[0])
#
#person = database.get_person_by_id(20)
#print(person)
#        
#person = database.get_latest_id()
#print(person)

#database = FaceRecognitionDataBase("face_db.db")
#
#persons = database.get_all_id()
#print(persons)
