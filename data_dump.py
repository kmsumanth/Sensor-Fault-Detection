import pandas as pd
import json
from sensor.config import mongo_client
from dotenv import load_dotenv
load_dotenv()

DATA_FILE_PATH="./notebook/aps_failure_training_set1.csv"
DATABASE_NAME='aps'
COLLECTION_NAME='sensor'

if __name__=='__main__':
    df=pd.read_csv(DATA_FILE_PATH)
    print(f"No of Rows and Columns :{df.shape}")

    # convert dataframe into json so that we can dump data into mongo db
    df.reset_index(drop=True,inplace=True)

    json_record=list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    #insert converted data into mongo db 

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
    print("Data inserted successfully into mongo db ")

