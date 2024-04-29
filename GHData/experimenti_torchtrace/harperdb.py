# -*- coding: utf-8 -*-
import numpy as np
import http.client as http
import json
import math
import urllib.request
import base64
import requests
import sys
import os
import time
import uuid
import checksize

DEFAULT_SCHEMA = 'torchtrace'
DEFAULT_TABLE = 'trace'
DEFAULT_HASH = 'id'
DEFAULT_URL = 'http://localhost:9925'
DEFAULT_USER = 'harperdb'
DEFAULT_PASSWORD = 'harperdb'
DEFAULT_HDB_PATH = '/data/hdc'


def connect(url=DEFAULT_URL, user=DEFAULT_USER, password=DEFAULT_PASSWORD):

    # validate that the server is up and running
    ping(url, user, password)

def ping(url=DEFAULT_URL, user=DEFAULT_USER, password=DEFAULT_PASSWORD):

    op_dict = {
        'operation': 'describe_all'
    }

    response_json = postToHarper(op_dict, url, 'harperdb', 'harperdb')

    return response_json

def showLogs():

	opt_dict = {
		"operation":"read_log",
		"limit":10,
		"start":0,
		"from":"2017-07-10",
		"until":"2019-07-11",
		"order":"asc"
	}

	printResponse(postToHarper(opt_dict, url=DEFAULT_URL, user=DEFAULT_USER, password=DEFAULT_PASSWORD))

def printResponse(response):
    pp_json(response)

def pp_json(response_json, sort=True, indents=4):
    if type(response_json) is str:
        print(json.dumps(json.loads(response_json), sort_keys=sort, indent=indents))
    else:
        print(json.dumps(response_json, sort_keys=sort, indent=indents))
    return None

def validateSchema(user=DEFAULT_USER, password=DEFAULT_PASSWORD, url=DEFAULT_URL, schema_name=DEFAULT_SCHEMA):

    op_dict = {
        'operation': 'create_schema',
        'schema': schema_name
    }
    postToHarper(op_dict, url, user=DEFAULT_USER, password=DEFAULT_PASSWORD)


def describeSchema(user=DEFAULT_USER, password=DEFAULT_PASSWORD, url=DEFAULT_URL, schema_name=DEFAULT_SCHEMA):

    op_dict = {
        'operation': 'describe_all'
    }

    printResponse(postToHarper(op_dict, url, user=DEFAULT_USER, password=DEFAULT_PASSWORD))

def createSchema(user=DEFAULT_USER, password=DEFAULT_PASSWORD, url=DEFAULT_URL, schema_name=DEFAULT_SCHEMA):

    op_dict = {
        'operation': 'create_schema',
        'schema': schema_name
    }

    print(postToHarper(op_dict, url, DEFAULT_USER, DEFAULT_PASSWORD))

def dropSchema(user=DEFAULT_USER, password=DEFAULT_PASSWORD, url=DEFAULT_URL, schema_name=DEFAULT_SCHEMA):

    op_dict = {
        'operation': 'drop_schema',
        'schema': schema_name
    }

    print(postToHarper(op_dict, url, user, password))


def createTable(table=DEFAULT_TABLE, schema=DEFAULT_SCHEMA, hash_value=DEFAULT_HASH):

    op_dict = {
        'operation': 'create_table',
        'schema': schema,
        'table': table,
        'hash_attribute': hash_value
    }

    print(postToHarper(op_dict))


def postToHarper(data, url="http://localhost:9925", user='harperdb', password='harperdb'):

    username_password_string = "{0}:{1}".format(user, password).encode()
    username_password_b64_encoded = base64.b64encode(username_password_string)

    encoded_username_password = 'Basic {0}'.format(
        username_password_b64_encoded.decode("utf-8"))

    headers = {
        'Content-Type': "application/json",
        'Authorization': encoded_username_password
    }

    response = requests.request(
        "POST", url, data=json.dumps(data), headers=headers)

    # print (data)
    print(response)
    return response.json()


def insert_narray_serialized(narray, label, schema=DEFAULT_SCHEMA, initialize_schema=False):

    size = sys.getsizeof(narray)
    dir_size = getDirectorySize(DEFAULT_HDB_PATH)

    print("Directory Size: ")
    print(dir_size)
    
    data = [
        {
            "id": uuid.uuid4().hex,
            "time_stamp": time.time(),
            "size": size,
            "size_on_disk": getDirectorySize(DEFAULT_HDB_PATH),
            "narray": narray.tolist()
        }
    ]

    op_dict = {
        'operation': 'insert',
        'schema': schema,
        'table': label,
        'records': data
    }

    print(postToHarper(op_dict))

## insert a numpy array as a harperdb table 
def insert_narray_x_y_fanned(narray, label, schema=DEFAULT_SCHEMA, initialize_schema=False):

    size = sys.getsizeof(narray)
    dir_size = getDirectorySize(DEFAULT_HDB_PATH)
    size_on_disk = getDirectorySize(DEFAULT_HDB_PATH)

    record_data = [] 

    for (k,v), value in np.ndenumerate(narray):
        data = { 
                "id": uuid.uuid4().hex,
                 "k{0}".format(str(k)) : value,
                "time_stamp": time.time(),
                "size": size,
                "size_on_disk": size_on_disk
            }

        record_data.append(data)

    op_dict = {
        "operation": 'insert',
        "schema": schema,
        "table": label,
        "records": record_data 
        }

    print(op_dict)
    print("Posting to HDB")
    op_size = sys.getsizeof(op_dict)
    print(op_size)
    print(postToHarper(op_dict))

def getDirectorySize(path=DEFAULT_HDB_PATH):

    return checksize.get_size(path=DEFAULT_HDB_PATH)

## insert a numpy array as a harperdb table 
def insert_narray_x_y(narray, label, schema=DEFAULT_SCHEMA, initialize_schema=False):

    size = sys.getsizeof(narray)
    dir_size = getDirectorySize(DEFAULT_HDB_PATH)

    size_on_disk = getDirectorySize(DEFAULT_HDB_PATH)

    record_data = [] 
    #For each K_V add a record!
    for (k,v), value in np.ndenumerate(narray):
        data = { 
                "id": uuid.uuid4().hex,
                "x": k,
                "y": v,
                "value": value,
                "time_stamp": time.time(),
                "size": size,
                "size_on_disk": size_on_disk
            }

        record_data.append(data)

    op_dict = {
        "operation": 'insert',
        "schema": schema,
        "table": label,
        "records": record_data 
        }

    # print(op_dict)
    print("Posting to HDB")
    op_size = sys.getsizeof(op_dict)
    print(postToHarper(op_dict))

def getDirectorySize(path=DEFAULT_HDB_PATH):

    return checksize.get_size(path=DEFAULT_HDB_PATH)

def exportResults(schema=DEFAULT_SCHEMA, table=DEFAULT_TABLE, results_dir='/home/john/code/torchtrace/results'):

    schema_table = schema + '.' + table
    query = "select time_stamp, size, size_on_disk from {0} order by time_stamp".format(schema_table) 
    
    print("Result Export Query: " + query)

    op_dict = {
        "operation": "export_local",
        "format": "csv",
        "path":  results_dir,
        "search_operation": {
            "operation": "sql",
                "sql": query 
    	}
    }	

    print(postToHarper(op_dict))

def exportTableToCSV(schema=DEFAULT_SCHEMA, table=DEFAULT_TABLE):

    schema_table = schema + '.' + table
    query = "select time, size, size_on_disk from {0}".format(schema_table) + " order by time"

    op_dict = {
        "operation": "export_local",
        "format": "csv",
        "path":  '/home/john/',
        "search_operation": {
            "operation": "sql",
                "sql": query 
        }
    }

    print(postToHarper(op_dict))

def batchInsertTensors(iterations=1, schema=DEFAULT_SCHEMA, table=DEFAULT_TABLE, user=DEFAULT_USER, password=DEFAULT_PASSWORD):

    for k in range(0, iterations):

        D_in, D_out = 10 + k, 10 + k

        # Create random input and output data
        x = np.random.randn(D_in, D_out)

        size = getDirectorySize()

        data = [
            {
                "id": uuid.uuid4().hex,
                "time": time.time(),
                "preinsert_size": size,
                "array_size_bytes": x.size * x.itemsize,
                "array": x.tolist()
            }
        ]

        op_dict = {
            'operation': 'insert',
            'schema': schema,
            'table': table,
            'records': data
        }

        print(postToHarper(op_dict))

