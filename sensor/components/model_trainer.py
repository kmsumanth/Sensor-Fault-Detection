import os , sys
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import artifact_entity,config_entity
from sensor import utils 


class Model_trainer:
    def __init__(self,
                 model_trainer_config:config_entity.ModelTrainingConfig,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact
                 ):
        try:
            logging.info(f"{'>>'*20} Model Training {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise SensorException(e,sys)
        
    def fine_tune(self):
        try:
            # write code for grid search cv
            pass
        except Exception as e:
            raise SensorException(e,sys)
        
    def train_model(self,X,y):
        try:
            xgb_classifier=XGBClassifier()
            xgb_classifier.fit(X,y)
            return xgb_classifier
        except Exception as e:
            raise SensorException(e,sys)
        
    def initiate_model_training(self)->artifact_entity.ModelTrainingArtifact:
        try:
            logging.info("Loading train and test array ")
            train_arr=utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr=utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            logging.info("splitting input and target column from both train and test ")
            X_train,y_train=train_arr[:,:-1],train_arr[:,-1]
            X_test,y_test=test_arr[:,:-1],test_arr[:,-1]

            logging.info("Training model")
            model=self.train_model(X_train,y_train)

            logging.info("Calculating f1 train score")
            yhat_train=model.predict(X_train)
            f1_train_score=f1_score(y_train,yhat_train)

            logging.info("Calculating f1 test score")
            yhat_test=model.predict(X_test)
            f1_test_score=f1_score(y_test,yhat_test)

            logging.info(f"train score:{f1_train_score} and tests score {f1_test_score}")

            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")
            
            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_thresold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")
            #save the trained model
            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainingArtifact(model_file_path=self.model_trainer_config.model_path, 
            f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e,sys)
        
    
       