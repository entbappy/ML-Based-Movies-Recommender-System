import os
import sys
import pickle
import numpy as np
from app_logger.logger import logging
from sklearn.metrics.pairwise import cosine_similarity
from app_exception.exception_handler import AppException
from app_configuration.configuration import AppConfiguration



class ModelTrainer:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.model_trainer_config = app_config.get_model_trainer_config()
        except Exception as e:
            raise AppException(e, sys) from e


    @staticmethod
    def load_numpy_array_data(file_path: str) -> np.array:
        """
        load numpy array data from file
        file_path: str location of file to load
        return: np.array data loaded
        """
        try:
            with open(file_path, 'rb') as file_obj:
                
                return np.load(file_obj)
        except Exception as e:
            raise AppException(e, sys) from e

    
    def train(self):
        try:
            #loading numpy array data
            vector = ModelTrainer.load_numpy_array_data(self.model_trainer_config.transformed_data_file_dir)
            #training model using cosine_similarity
            similarity = cosine_similarity(vector)
            logging.info(f" Similarity vector shape: {similarity.shape}")

            #Saving similarity object for recommendations
            os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)
            file_name = os.path.join(self.model_trainer_config.trained_model_dir,self.model_trainer_config.trained_model_name)
            pickle.dump(similarity,open(file_name,'wb'))
            logging.info(f"Saving final similarity model to {file_name}")

        except Exception as e:
            raise AppException(e, sys) from e

    

    def initiate_model_trainer(self):
        try:
            logging.info(f"{'='*20}Model Trainer log started.{'='*20} ")
            self.train()
            logging.info(f"{'='*20}Model Trainer log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e
