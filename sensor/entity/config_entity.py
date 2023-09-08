import os
import sys
from datetime import datetime
from sensor.logger import logging
from sensor.exception import SensorException

FILE_NAME='sensor.csv'
TRAIN_FILE_NAME='train.csv'
TEST_FILE_NAME='test.csv'
TRANSFORMER_FILE_NAME='transformer.pkl'
TARGET_ENCODER_OBJECT_FILE_NAME='target_encoder.pkl'
MODEL_FILE_NAME='model.pkl'

class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir=os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m-%d-%Y_%H-%M')}")
        except Exception as e:
            raise SensorException(e,sys)
        
class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name='aps'
            self.collection_name='sensor'
            self.data_ingestion_dir=os.path.join(training_pipeline_config.artifact_dir,"data_ingestion")
            self.feature_store_file_path=os.path.join(self.data_ingestion_dir,'feature_store',FILE_NAME)
            self.train_file_path=os.path.join(self.data_ingestion_dir,'dataset',TRAIN_FILE_NAME)
            self.test_file_path=os.path.join(self.data_ingestion_dir,'dataset',TEST_FILE_NAME)
            self.test_size=0.2
        except Exception as e:
            SensorException(e,sys)

    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception  as e:
            raise SensorException(e,sys)     


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir=os.path.join(training_pipeline_config.artifact_dir,'data_validation')
        self.report_file_path=os.path.join(self.data_validation_dir,'report.yaml')
        self.missing_value_thresold=0.2
        self.base_file_path=os.path.join("./notebook/aps_failure_training_set1.csv")


class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir=os.path.join(training_pipeline_config.artifact_dir,"Data_transformation")
        self.transformer_object_path=os.path.join(self.data_transformation_dir,"Transformer",TRANSFORMER_FILE_NAME)
        self.transformed_train_path=os.path.join(self.data_transformation_dir,'transformed',TRAIN_FILE_NAME.replace('csv','npz'))
        self.transformed_test_path=os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME.replace('csv','npz'))
        self.target_encoder_path=os.path.join(self.data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)



class ModelTrainingConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_training_dir=os.path.join(training_pipeline_config.artifact_dir,"Model_training")
        self.model_path=os.path.join(self.model_training_dir,"model",MODEL_FILE_NAME)
        self.expected_score=0.7
        self.overfitting_thresold=0.1

        
class ModelEvaluationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.change_thresold=0.01

class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_pusher_dir=os.path.join(training_pipeline_config.artifact_dir,"model_pusher")
        self.saved_model_dir=os.path.join("saved_models")
        self.pusher_model_dir=os.path.join(self.model_pusher_dir,"saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir,MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir,TRANSFORMER_FILE_NAME)
        self.pusher_target_encoder_path = os.path.join(self.pusher_model_dir,TARGET_ENCODER_OBJECT_FILE_NAME)
        
    
        

        

