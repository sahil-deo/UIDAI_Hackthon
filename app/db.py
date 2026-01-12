import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()


DB_CONFIG = {

    "host": os.getenv('HOST'),
    "port": os.getenv('PORT'),
    "dbname": os.getenv('DBNAME'),
    "user": os.getenv('DBUSER'),
    "password": os.getenv('PASSWORD'),


}


def get_conn():
    # centralized DB connection helper
    return psycopg2.connect(**DB_CONFIG)