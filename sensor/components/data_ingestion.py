import os
import sys
import pandas as pd
import numpy as np

from sensor.logger import logging
from sensor.exception import SensorException
from sensor import utils
from sensor.entity import config_entity
from sensor.entity import artifact_entity

from sklearn.model_selection import train_test_split

from sensor.entity.config_entity import TrainingPipelineConfig
from sensor.entity.config_entity import DataIngestionConfig


class DataIngestion:

    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data ingestion {'<<'*20}")
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise SensorException(e,sys)
        
    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info("Exporting collection as dataframe")
            df:pd.DataFrame=utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name
            )

            logging.info("Saving data in feature store ")

            # replace na values with nan

            df.replace("na",np.nan,inplace=True)

            feature_store_dir=os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            logging.info("Saving df to feature store folder ")
            df.to_csv(self.data_ingestion_config.feature_store_file_path,index=False,header=True)

            logging.info("Splitting data into train and test ")
            train_df,test_df=train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)
            
            logging.info("Create dataset dirctory if not available ")
            dataset_dir=os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            train_df.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)

            # Prepare artifact

            data_ingestion_artifact=artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

            logging.info(f"DataIngestion artifact {data_ingestion_artifact}")

            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e,sys)
        
        



