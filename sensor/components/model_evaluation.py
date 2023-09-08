import os , sys
import pandas as pd
from sklearn.metrics import f1_score

from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import artifact_entity,config_entity
from sensor.config import TARGET_COLUMN
from sensor.utils import load_object
from sensor.predictor import ModelResolver


class ModelEvaluation:
    def __init__(self,
                 model_eval_config:config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifact_entity.ModelTrainingArtifact
                 ):
        try:
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise SensorException(e,sys)
        
    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            #if saved model folder has model then we will compare 
            #which model is good 
            logging.info("If saved model folder has model then we will compare")
            #get latest model path 
            latest_dir_path=self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact=artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,improved_accuracy=None)
                return model_eval_artifact
            
            #finding the location of transformer model and target encoder 
            logging.info("Finding the location of transformer model and targer encdoer ")
            transformer_path=self.ModelResolver.get_latest_dir_path()
            model_path=self.ModelResolver.get_latest_dir_path()
            target_encoder_path=self.ModelResolver.get_latest_dir_path()
            
            logging.info("Previously trained objects of model , tranformer , target encoder")
            model=load_object(model_path)
            transformer=load_object(transformer_path)
            target_encoder=load_object(target_encoder_path)

            logging.info("Currently Trained model objects")
            current_transformer=load_object(self.data_transformation_artifact.transformer_obj_path)
            current_model=load_object(self.model_trainer_artifact.model_file_path)
            current_target_encoder=load_object(self.data_transformation_artifact.target_encoder_file_path)

            test_df=self.data_ingestion_artifact.test_file_path
            target_df=test_df[TARGET_COLUMN]
            y_true=target_encoder.transform(target_df)

            input_feature_name=list(transformer.feature_names_in_)
            input_arr=transformer.transform(test_df[input_feature_name])
            y_pred=model.predict(input_arr)
            print(f"Prediction using previous model : {target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score=f1_score(y_pred=y_pred,y_true=y_true)
            logging.info(f"Accuracy of previous trained model : {previous_model_score}")

            #accuracy using current model 

            input_feature_name=list(current_model.feature_names_in_)
            input_arr=current_transformer.transform(test_df[input_feature_name])
            y_pred=current_model.predict(input_arr)
            y_true=current_target_encoder.transform(test_df[TARGET_COLUMN])
            print(f"Prediction using current trained model : {target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score=f1_score(y_pred=y_pred,y_true=y_true)
            logging.info(f"Accuracy of current model is :{current_model_score}")

            if current_model_score<=previous_model_score:
                logging.info("Current trained model is not better than previous model ")
                raise Exception("Current trained model is not better than previous model ")
            
            model_eval_artifact=artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                        improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"Model evaluation Artifact {model_eval_artifact}")
            return model_eval_artifact


        except Exception as e:
            raise SensorException(e,sys)
        