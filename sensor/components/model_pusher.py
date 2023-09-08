import os , sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity,artifact_entity
from sensor.predictor import ModelResolver
from sensor.utils import save_object,load_object
from sensor.entity.artifact_entity import DataTransformationArtifact,ModelTrainingArtifact,ModelPusherArtifact

class ModelPusher:
    def __init__(self,
                 model_pusher_config:config_entity.ModelPusherConfig,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_artifact:ModelTrainingArtifact
                 ):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config=model_pusher_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver=ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise SensorException(e,sys)
        
    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            #load objects 
            logging.info("Load transfomer model and target encoder ")
            model=load_object(file_path=self.model_trainer_artifact.model_file_path)
            transformer=load_object(file_path=self.data_transformation_artifact.transformer_obj_path)
            target_encoder=load_object(file_path=self.data_transformation_artifact.target_encoder_file_path)

            #model pusher dir

            logging.info("Saving model into model pusher directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path,obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path,obj=target_encoder)
            save_object(file_path=self.model_pusher_config.pusher_model_path,obj=model)
            

            #saved model dir
            logging.info("Saving model in saved model dir")
            transformer_path=self.model_resolver.get_latest_save_transformer_path()
            model_path=self.model_resolver.get_latest_save_model_path()
            target_encoder_path=self.model_resolver.get_latest_save_target_encoder_path()

            save_object(file_path=transformer_path,obj=transformer_path)
            save_object(file_path=model_path,obj=model)
            save_object(file_path=target_encoder_path,obj=target_encoder)

            model_pusher_artifact=artifact_entity.ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                saved_model_dir=self.model_pusher_config.saved_model_dir
            )
            logging.info(f"Model artifact : {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise SensorException(e,sys)
        