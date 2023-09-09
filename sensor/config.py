import pymongo
import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass
class EnvironmentVariable:
    mongo_db_url:str=os.getenv('MONGO_DB_URL')




env_var=EnvironmentVariable()
mongo_client=pymongo.MongoClient(env_var.mongo_db_url)
print("Connection Estalished successfully")



TARGET_COLUMN='class'