import os
import sys
import pandas as pd
import numpy as np
from movies_recommender_app_logger.logger import logging
from sklearn.feature_extraction.text import CountVectorizer
from movies_recommender_app_exception.exception_handler import AppException
from movies_recommender_app_configuration.configuration import AppConfiguration


class DataTransformation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_transformation_config = app_config.get_data_transformation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    
    @staticmethod
    def save_numpy_array_data(file_path: str, array: np.array):
        """
        Save numpy array data to file
        file_path: str location of file to save
        array: np.array data to save
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, 'wb') as file_obj:
                np.save(file_obj, array)
        except Exception as e:
            raise AppException(e, sys) from e

    
    def get_data_transformer(self):
        try:
            df = pd.read_csv(self.data_transformation_config.clean_data_file_path)
            cv = CountVectorizer(max_features=5000,stop_words='english')
            vector = cv.fit_transform(df['tags']).toarray()
            logging.info(f" Shape of the final vector: {vector.shape}")

            #saving vector as numpy array for training
            DataTransformation.save_numpy_array_data(file_path=os.path.join(self.data_transformation_config.transformed_data_dir,"transformed_data.npy"),array=vector)
            logging.info(f"Saving final vector as numpy array to {self.data_transformation_config.transformed_data_dir}")

        except Exception as e:
            raise AppException(e, sys) from e

    

    def initiate_data_transformation(self):
        try:
            logging.info(f"{'='*20}Data Transformation log started.{'='*20} ")
            self.get_data_transformer()
            logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e



