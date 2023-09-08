from sensor.pipeline.training import start_training_pipeline
from sensor.pipeline.batch_prediction import start_batch_prediction

print(__name__)

if __name__=="__main__":
    try:
        #start training 
        start_training_pipeline()
    except Exception as e:
        print(e)