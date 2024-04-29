# -*- coding: utf-8 -*-
"""...
"""
from sqlalchemy import *
import psycopg2 as psycopg

# Create a connection to a PostGreSQL database
engine = create_engine('postgres://alxfed:postgres@localhost:5432/pg_db')

def connect_pg():
    return psycopg.connect(user='rick', host='localhost')

enging = create_engine('postgres://', creator=connect_pg)


def main():
    # Create a connection to a PostGreSQL database
    engine = create_engine('postgres://alxfed:postgres@localhost:5432/pg_db')
    print('ok')
    return


if __name__ == '__main__':
    main()
    print('\ndone')