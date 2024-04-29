import unittest
import arraytojson
import numpy
import json
import harperdb as hdb

DEFAULT_SCHEMA = 'unittest'
DEFAULT_TABLE = 'unittest'
DEFAULT_HASH = 'id'
DEFAULT_URL = 'http://localhost:9925'
DEFAULT_USER = 'harperdb'
DEFAULT_PASSWORD = 'harperdb'
DEFAULT_HDB_PATH = '/hdb/john_hdb'

class HarperTests(unittest.TestCase):

     #Clear our test schema
    def setUp(self):
       result = hdb.dropSchema(user=DEFAULT_USER, password=DEFAULT_PASSWORD, url=DEFAULT_URL, schema_name=DEFAULT_SCHEMA)

    def test_ping(self):
       result = hdb.ping()
       self.assertTrue(result)

    def test_create_schema(self):
        result = hdb.createSchema(user=DEFAULT_USER, password=DEFAULT_PASSWORD, url=DEFAULT_URL, schema_name=DEFAULT_SCHEMA)

    def test_dimension_scale(self):
        #10x10 matrix
        dimensions = 10
        result = arraytojson.numpyArrayToJson(10)
        self.assertTrue(len(result)==9)

if __name__ == '__main__':
    unittest.main()