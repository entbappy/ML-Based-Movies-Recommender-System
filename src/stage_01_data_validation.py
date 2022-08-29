import os
import sys
import ast 
import pandas as pd
import nltk
import pickle
from nltk.stem import PorterStemmer
from app_logger.logger import logging
from app_exception.exception_handler import AppException
from app_configuration.configuration import AppConfiguration


class DataValidation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_validation_config= app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    @staticmethod
    def convert_str_to_list(text):
        '''
        for converting str to list
        '''
        L = []
        for i in ast.literal_eval(text):
            L.append(i['name']) 
        return L

    @staticmethod
    def convert_cast(text):
        '''
        This function will return top 3 cast from all
        '''
        L = []
        counter = 0
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
            counter+=1
        return L

    @staticmethod
    def fetch_director(text):
        '''
        This function will only director
        '''
        L = []
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    
    @staticmethod
    def remove_space(L):
        '''
        This function will remove spaces
        Examples: 'Bappy Ahmed' to 'BappyAhmed'      
        '''
        L1 = []
        for i in L:
            L1.append(i.replace(" ",""))
        return L1

    
    @staticmethod
    def stems(text):
        '''
        This function will steam the texts  
        '''
        T = []
        ps = PorterStemmer()
        for i in text.split():
            T.append(ps.stem(i))
        
        return " ".join(T)

    
    def preprocess_data(self):
        try:
            movies = pd.read_csv(self.data_validation_config.movies_csv_file)
            credits = pd.read_csv(self.data_validation_config.credit_csv_file)
            
            logging.info(f" Shape of movies data file: {movies.shape}")
            logging.info(f" Shape of credits data file: {credits.shape}")
            
            # Merging credits with movies based on title 
            movies = movies.merge(credits,on='title')
            # logging.info(f" Shape of final dataset after merging: {movies.shape}")

            # Keeping important columns for recommendation
            movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
            #dropping missing values
            movies.dropna(inplace=True)

            # converting genres columns to list data types
            movies['genres'] = movies['genres'].apply(DataValidation.convert_str_to_list)
            # converting keywords columns to list data types
            movies['keywords'] = movies['keywords'].apply(DataValidation.convert_str_to_list)
            # Here i am just keeping top 3 cast
            movies['cast'] = movies['cast'].apply(DataValidation.convert_cast)
            # Here i am just keeping director job title
            movies['crew'] = movies['crew'].apply(DataValidation.fetch_director)
            # handle overview (converting to list)
            movies['overview'] = movies['overview'].apply(lambda x:x.split())

            # now removing spaces for below columns
            movies['cast'] = movies['cast'].apply(DataValidation.remove_space)
            movies['crew'] = movies['crew'].apply(DataValidation.remove_space)
            movies['genres'] = movies['genres'].apply(DataValidation.remove_space)
            movies['keywords'] = movies['keywords'].apply(DataValidation.remove_space)

            # Concatinating all columns as tags
            movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

            # droping those extra columns
            new_df = movies[['movie_id','title','tags']]
            # Converting list to str
            new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
            # Converting to lower case
            new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
            #steaming text
            new_df['tags'] = new_df['tags'].apply(DataValidation.stems)
            logging.info(f" Shape of final data: {new_df.shape}")

            # Saving the cleaned data for transformation
            os.makedirs(self.data_validation_config.clean_data_dir, exist_ok=True)
            new_df.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'clean_data.csv'), index = False)
            logging.info(f"Saved cleaned data to {self.data_validation_config.clean_data_dir}")


            #saving new_df objects for web app
            new_df_serialization = movies[['movie_id','title']]
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            pickle.dump(new_df_serialization,open(os.path.join(self.data_validation_config.serialized_objects_dir, "movie_list.pkl"),'wb'))
            logging.info(f"Saved serialization object to {self.data_validation_config.serialized_objects_dir}")

        except Exception as e:
            raise AppException(e, sys) from e

    
    def initiate_data_validation(self):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*20} ")
            self.preprocess_data()
            logging.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e



    