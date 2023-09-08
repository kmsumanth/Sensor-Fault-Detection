import os , sys
import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import ks_2samp

from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity,artifact_entity
from sensor import utils
from sensor.config import TARGET_COLUMN


class DataValidation:
    def __init__(self,
                 data_validation_config:config_entity.DataValidationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact
                 ) :
        try:
            logging.info(f"{'>>'*20}Data validation {'<<'*20}")
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.validation_error=dict()
        except Exception as e:
            raise SensorException(e,sys)
        
    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop column which contains missing values more than the specified thresold 
        ==============================================================================================
        Params : 
        df : Dataframe 
        thresold : Percentage criteria to drop a column 
        ==============================================================================================
        returns pandas dataframe if column is present after dropping else nothing  
        """

        try:
            thresold=self.data_validation_config.missing_value_thresold
            null_report=df.isna().sum()/df.shape[0]
            # select column names having null values greater than the thresold 
            logging.info("Selectiong column names having null values greater than thresold ")
            drop_column_names=null_report[null_report>thresold].index

            logging.info(f"Columns to drop {list(drop_column_names)}")
            self.validation_error[report_key_name]=list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            #return none if df is empty

            if len(df.columns)==0:
                return None
            else:
                return df


        except Exception as e:
            raise SensorException(e,sys)


    def is_required_columns_exits(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
            base_columns=base_df.columns
            current_columns=current_df.columns

            missing_columns=[]

            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"{base_column} not available ")
                    missing_columns.append(base_column)
            
            if(len(missing_columns)>0):
                self.validation_error[report_key_name]=missing_columns
                return False
            return True

        except Exception as e:
            raise SensorException(e,sys)
        
    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report=dict()

            base_columns=base_df.columns
            current_columns=current_df.columns

            for base_column in base_columns:
                base_data,current_data=base_df[base_column],current_df[base_column]
                #null hypothesis is that both the data is drawn from same distribution 

                logging.info(f"hypothesis {base_column}:{base_data}, {current_data}")

                same_distribution=ks_2samp(base_data,current_data)

                if same_distribution.pvalue > 0.05:
                    # we accept null hypothesis
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":True
                    }

                else:
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":False
                    }


            self.validation_error[report_key_name]=drift_report

        except Exception as e:
            raise SensorException(e,sys)
        
    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
             logging.info("Reading base Dataframe")
             base_df=pd.read_csv(self.data_validation_config.base_file_path)
             base_df.replace("na",np.NaN,inplace=True)
             logging.info("Replace na values in dataframe base_df")
             # df has na as null 
             logging.info("Drop null values columns from base_df")
             base_df=self.drop_missing_values_columns(df=base_df,report_key_name="Missing_value_within_base_dataframe")
             
             logging.info("Reading train dataframe")
             train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
             logging.info("Reading test dataframe")
             test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)

             logging.info("Drop null columns from train dataframe")
             train_df=self.drop_missing_values_columns(train_df,report_key_name="missing_value-within_train_dataframe")
             logging.info("Drop null columns from test dataframe")
             test_df=self.drop_missing_values_columns(test_df,report_key_name='missing_values_within_test_dataframe')

             exclude_columns=[TARGET_COLUMN]
             base_df = utils.convert_columns_float(df=base_df, exclude_columns=exclude_columns)
             train_df = utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
             test_df = utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)

             logging.info(f"Is all required columns present in train df")
             train_df_columns_status =self.is_required_columns_exits(base_df=base_df,current_df=train_df,report_key_name="missing_columns_within_train_dataset")
             logging.info(f"Is all required columns present in test df")
             test_df_columns_status =self.is_required_columns_exits(base_df=base_df,current_df=test_df,report_key_name="missing_columns_within_test_dataset")
            
             if train_df_columns_status:
                logging.info(f"As all column are available in train df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df,report_key_name="data_drift_within_train_dataset")
             if test_df_columns_status:
                logging.info(f"As all column are available in test df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df,report_key_name="data_drift_within_test_dataset")
             
             #write the report 

             logging.info("Writing the report in yaml file")
             utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,data=self.validation_error)

             data_validation_artifact=artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
             logging.info(f"data validation artifact {data_validation_artifact}")
             return data_validation_artifact

        except Exception as e:
            raise SensorException(e,sys)