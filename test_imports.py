import sys
import config
from advanced_data_pipeline import AgroScoreInferencePipeline
from advanced_ml_model import AgroScoreModel
from advanced_app import load_data, initialize_data_pipeline

print("Imports successful!")
pipeline = initialize_data_pipeline()
print("Pipeline initialized!")
